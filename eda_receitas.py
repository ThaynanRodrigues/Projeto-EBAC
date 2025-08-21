import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # usar backend não interativo p/ salvar figuras
import matplotlib.pyplot as plt


# ==========================
# CONFIG
# ==========================
PASTA = "C:\\Users\\thayn\\OneDrive\\Área de Trabalho\\CSV TRATADOS"  # pasta onde estão os CSVs tratados
ARQS_TRATADOS = [
    "2023_Receitas_com_duas_novas_colunas.csv",
    "2024_Receitas_com_duas_novas_colunas.csv",
    "2025_Receitas_com_duas_novas_colunas.csv",
]
OUT_DIR = os.path.join(PASTA, "saida_eda")
RELATORIO_HTML = os.path.join(OUT_DIR, "Relatorio_EDA_Receitas_2023_2025.html")

os.makedirs(OUT_DIR, exist_ok=True)


# ==========================
# Funções utilitárias
# ==========================
def carregar_bases(caminhos, pasta):
    frames = []
    for caminho in caminhos:
        p = os.path.join(pasta, caminho)
        if not os.path.exists(p):
            print(f"[AVISO] Não encontrei: {p}")
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        # inferir Ano pelo nome do arquivo
        ano_inferido = None
        for ano in [2023, 2024, 2025]:
            if str(ano) in caminho:
                ano_inferido = ano
                break
        if "Ano" not in df.columns and ano_inferido is not None:
            df["Ano"] = ano_inferido
        frames.append(df)
    if not frames:
        raise FileNotFoundError("Nenhuma base tratada encontrada. Verifique os caminhos em ARQS_TRATADOS.")
    return pd.concat(frames, ignore_index=True)


def garantir_tipos(dados: pd.DataFrame) -> pd.DataFrame:
    # numéricos
    for c in ["VALOR PREVISTO ATUALIZADO", "VALOR REALIZADO", "DIFERENÇA PREVISTO_REALIZADO"]:
        if c in dados.columns:
            dados[c] = pd.to_numeric(dados[c], errors="coerce")

    # datas
    if "DATA LANÇAMENTO (pad)" in dados.columns:
        dados["DATA LANÇAMENTO"] = pd.to_datetime(dados["DATA LANÇAMENTO (pad)"], errors="coerce")
    elif "DATA LANÇAMENTO" in dados.columns:
        dados["DATA LANÇAMENTO"] = pd.to_datetime(dados["DATA LANÇAMENTO"], errors="coerce")
    else:
        dados["DATA LANÇAMENTO"] = pd.NaT

    # mês/ano
    if "MÊS_ANO LANÇAMENTO" in dados.columns:
        dados["MÊS_ANO LANÇAMENTO"] = dados["MÊS_ANO LANÇAMENTO"].astype(str)
    else:
        dados["MÊS_ANO LANÇAMENTO"] = dados["DATA LANÇAMENTO"].dt.to_period("M").astype(str)

    return dados


def salvar_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def tabela_para_csv(df, nome):
    p = os.path.join(OUT_DIR, nome)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


# ==========================
# Carregar e preparar
# ==========================
dados = carregar_bases(ARQS_TRATADOS, PASTA)
dados = garantir_tipos(dados)

# Colunas de texto úteis (podem existir ou não)
CAT_COL = "CATEGORIA ECONÔMICA" if "CATEGORIA ECONÔMICA" in dados.columns else (
          "NOME ÓRGÃO" if "NOME ÓRGÃO" in dados.columns else None)
UG_COL  = "NOME UNIDADE GESTORA" if "NOME UNIDADE GESTORA" in dados.columns else (
          "NOME ÓRGÃO" if "NOME ÓRGÃO" in dados.columns else None)

# ==========================
# (1) Distribuição: VALOR REALIZADO por ano (boxplot)
# ==========================
fig1 = os.path.join(OUT_DIR, "eda1_boxplot_realizado_por_ano.png")
plt.figure()
dados.boxplot(column="VALOR REALIZADO", by="Ano")
plt.title("Distribuição do Valor Realizado por Ano")
plt.suptitle("")
plt.xlabel("Ano")
plt.ylabel("Valor Realizado")
salvar_fig(fig1)

resumo_dist = dados.groupby("Ano")["VALOR REALIZADO"].describe().round(2)
tabela_para_csv(resumo_dist.reset_index(), "tabela_1_resumo_distribuicao_por_ano.csv")


# ==========================
# (2) Média/Mediana da diferença por categoria (Top 10 por média absoluta)
# ==========================
if CAT_COL is None:
    CAT_COL = dados.columns[0]  # fallback para algum texto

diff_stats = (
    dados.groupby([CAT_COL, "Ano"])["DIFERENÇA PREVISTO_REALIZADO"]
         .agg(media="mean", mediana="median", qtd="count")
         .round(2)
         .reset_index()
)
tabela_para_csv(diff_stats, "tabela_2_media_mediana_diferenca_por_categoria_ano.csv")

top_media = (
    diff_stats.groupby(CAT_COL)["media"]
              .mean().abs().sort_values(ascending=False).head(10)
)
fig2 = os.path.join(OUT_DIR, "eda2_barras_media_diferenca_top10.png")
plt.figure()
top_media.plot(kind="bar")
plt.title("Top 10 Categorias por Média Absoluta da Diferença (Previsto vs Realizado)")
plt.xlabel(CAT_COL)
plt.ylabel("Média da Diferença")
salvar_fig(fig2)

tabela_para_csv(top_media.reset_index(name="media_abs"), "tabela_2b_top10_categorias_media_abs.csv")


# ==========================
# (3) Top 10 Unidades Gestoras por Valor Realizado (soma total)
# ==========================
if UG_COL is None:
    UG_COL = dados.columns[0]

top_ug_total = (
    dados.groupby(UG_COL)["VALOR REALIZADO"].sum().sort_values(ascending=False).head(10)
)
fig3 = os.path.join(OUT_DIR, "eda3_barras_top10_ug.png")
plt.figure()
top_ug_total.plot(kind="bar")
plt.title("Top 10 Unidades Gestoras por Valor Realizado (Total 2023–2025)")
plt.xlabel(UG_COL)
plt.ylabel("Soma do Valor Realizado")
salvar_fig(fig3)

tabela_para_csv(top_ug_total.reset_index(name="realizado_total"), "tabela_3_top10_ug_realizado_total.csv")


# ==========================
# (4) Dispersão Previsto × Realizado + correlação
# ==========================
fig4 = os.path.join(OUT_DIR, "eda4_scatter_previsto_vs_realizado.png")
plt.figure()
plt.scatter(dados["VALOR PREVISTO ATUALIZADO"], dados["VALOR REALIZADO"], s=8)
plt.title("Dispersão: Valor Previsto vs Valor Realizado (2023–2025)")
plt.xlabel("Valor Previsto Atualizado")
plt.ylabel("Valor Realizado")
salvar_fig(fig4)

corr_geral = float(dados[["VALOR PREVISTO ATUALIZADO", "VALOR REALIZADO"]]
                   .corr(method="pearson").iloc[0, 1])
corr_por_ano = (
    dados.groupby("Ano")[["VALOR PREVISTO ATUALIZADO", "VALOR REALIZADO"]]
         .corr(method="pearson")
         .reset_index()
         .rename(columns={"level_1": "Variável"})
)
tabela_para_csv(pd.DataFrame({"Correlação (geral)":[corr_geral]}), "tabela_4_correlacao_geral.csv")
tabela_para_csv(corr_por_ano, "tabela_4b_correlacao_por_ano.csv")

# (Opcional) Heatmap simples de correlação entre numéricos principais
fig4b = os.path.join(OUT_DIR, "eda4b_heatmap_correlacoes.png")
num_cols = ["VALOR PREVISTO ATUALIZADO", "VALOR REALIZADO", "DIFERENÇA PREVISTO_REALIZADO"]
num_cols = [c for c in num_cols if c in dados.columns]
if len(num_cols) >= 2:
    C = dados[num_cols].corr()
    plt.figure()
    im = plt.imshow(C, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlação entre Variáveis Numéricas")
    salvar_fig(fig4b)


# ==========================
# (5) Outliers na diferença por ano (IQR + boxplot)
# ==========================
fig5 = os.path.join(OUT_DIR, "eda5_boxplot_diferenca_por_ano.png")
plt.figure()
dados.boxplot(column="DIFERENÇA PREVISTO_REALIZADO", by="Ano")
plt.title("Diferença Previsto vs Realizado por Ano (IQR/Outliers)")
plt.suptitle("")
plt.xlabel("Ano")
plt.ylabel("Diferença (Realizado − Previsto)")
salvar_fig(fig5)

# Tabela de outliers usando regra IQR
outliers_list = []
for ano, g in dados.groupby("Ano"):
    s = g["DIFERENÇA PREVISTO_REALIZADO"].dropna()
    if len(s) < 5:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    out = g[(g["DIFERENÇA PREVISTO_REALIZADO"] < lim_inf) | (g["DIFERENÇA PREVISTO_REALIZADO"] > lim_sup)].copy()
    out["LIM_INF"], out["LIM_SUP"] = lim_inf, lim_sup
    outliers_list.append(out)

outliers_df = pd.concat(outliers_list, ignore_index=True) if outliers_list else pd.DataFrame()
if not outliers_df.empty:
    cols_show = [c for c in [UG_COL, "Ano", "VALOR PREVISTO ATUALIZADO", "VALOR REALIZADO",
                             "DIFERENÇA PREVISTO_REALIZADO", "MÊS_ANO LANÇAMENTO", CAT_COL]
                 if c in outliers_df.columns]
    tabela_para_csv(outliers_df[cols_show], "tabela_5_outliers_diferenca.csv")


# ==========================
# (6) Tendência mensal (extra)
# ==========================
trend = dados.groupby(["MÊS_ANO LANÇAMENTO"])["VALOR REALIZADO"].sum().reset_index()
try:
    trend["MES"] = pd.to_datetime(trend["MÊS_ANO LANÇAMENTO"] + "-01", errors="coerce")
    trend = trend.sort_values("MES")
except Exception:
    pass
fig6 = os.path.join(OUT_DIR, "eda6_linha_tendencia_mensal.png")
plt.figure()
plt.plot(trend["MES"], trend["VALOR REALIZADO"])
plt.title("Tendência Mensal do Valor Realizado (2023–2025)")
plt.xlabel("Mês")
plt.ylabel("Valor Realizado (Soma)")
salvar_fig(fig6)
tabela_para_csv(trend[["MÊS_ANO LANÇAMENTO", "VALOR REALIZADO"]], "tabela_6_tendencia_mensal.csv")


# ==========================
# Relatório HTML com principais achados
# ==========================
def img(path):
    return f'<img src="{os.path.basename(path)}" style="max-width:100%;height:auto;" />'

html = f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Relatório EDA – Receitas 2023–2025</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; }}
h2 {{ margin-top: 1.4rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
small {{ color: #555; }}
</style>
</head>
<body>
<h1>Relatório de Análise Exploratória – Receitas (2023–2025)</h1>
<small>Gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")}</small>

<h2>1) Distribuição do Valor Realizado por Ano</h2>
{img(fig1)}

<h2>2) Diferença Previsto vs Realizado por Categoria (Top 10)</h2>
{img(fig2)}

<h2>3) Top 10 Unidades Gestoras por Valor Realizado</h2>
{img(fig3)}

<h2>4) Dispersão Previsto vs Realizado e Correlação</h2>
<p>Correlação (Pearson) geral: <b>{np.round(corr_geral,3) if not np.isnan(corr_geral) else "N/A"}</b></p>
{img(fig4)}
{"<h3>Heatmap de correlações</h3>"+img(fig4b) if os.path.exists(fig4b) else ""}

<h2>5) Outliers de Diferença (IQR)</h2>
{img(fig5)}
{"<p>Arquivo com casos outliers: <code>tabela_5_outliers_diferenca.csv</code></p>" if not outliers_df.empty else "<p>Não foram identificados outliers relevantes (ou amostra insuficiente por ano).</p>"}

<h2>6) Tendência Mensal do Valor Realizado</h2>
{img(fig6)}

</body>
</html>
"""

with open(RELATORIO_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("\n=== EDA concluída! ===")
print(f"Saída: {OUT_DIR}")
print(f"- Relatório HTML: {RELATORIO_HTML}")
print("- Gráficos: eda1..eda6_*.png")
print("- Tabelas CSV: tabela_*.csv")