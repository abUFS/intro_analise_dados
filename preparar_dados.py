"""
Prepara os dados do Dia 2 a partir do dataset público da Olist (Kaggle: olistbr/brazilian-ecommerce).

Uso:
    python preparar_dados.py <pasta_com_csvs_da_olist> <pasta_de_saida>

Gera, para as versões "short" (amostra de 8.000 pedidos) e "full" (todos os pedidos):
    pedidos_sujo_<v>.csv    tabela de pedidos com problemas plantados (tíquetes 1 e 2)
    pedidos_limpo_<v>.csv   a mesma tabela já limpa (checkpoint a partir do tíquete 3)
    avaliacoes_<v>.csv      avaliações dos clientes, em tabela separada (tíquete 5)
    respostas_<v>.json      hashes das respostas, usados pela função conferir() do notebook
    codigos_<v>.md          gabarito com os valores e os códigos (SÓ PARA OS MONITORES, não publicar)

Fonte dos dados: Olist, licença CC BY-NC-SA 4.0.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SAL = "balcao-2026"
TAMANHO_SHORT = 8000
SEMENTE = 42
MIN_PEDIDOS = {"short": 30, "full": 300}

STATUS = {"delivered": "entregue", "shipped": "enviado", "canceled": "cancelado",
          "unavailable": "indisponivel", "invoiced": "faturado", "processing": "processando",
          "created": "criado", "approved": "aprovado"}
PAGAMENTO = {"credit_card": "cartao_credito", "boleto": "boleto", "voucher": "voucher",
             "debit_card": "cartao_debito", "not_defined": "nao_definido"}


# ------------------------------------------------------------------ conferência
def normalizar(valor, casas):
    if hasattr(valor, "strftime"):
        return valor.strftime("%Y-%m-%d")
    if isinstance(valor, str):
        v = valor.strip().lower()
        return v[:10] if len(v) >= 10 and v[4] == "-" and v[7] == "-" else v
    return f"{round(float(valor), casas):.{casas}f}"


def hash_resposta(texto):
    return hashlib.sha256((SAL + texto).encode()).hexdigest()


def codigo(texto):
    h = hashlib.sha256(("codigo" + SAL + texto).encode()).digest()
    cons, vog = "BCDFGKLMNPRSTVZ", "AEIOU"
    return "".join((cons if i % 2 == 0 else vog)[h[i] % (15 if i % 2 == 0 else 5)] for i in range(5))


# ------------------------------------------------------------------ base limpa
def montar_base(pasta):
    p = Path(pasta)
    orders = pd.read_csv(p / "olist_orders_dataset.csv")
    items = pd.read_csv(p / "olist_order_items_dataset.csv")
    pays = pd.read_csv(p / "olist_order_payments_dataset.csv")
    custs = pd.read_csv(p / "olist_customers_dataset.csv")
    prods = pd.read_csv(p / "olist_products_dataset.csv")
    revs = pd.read_csv(p / "olist_order_reviews_dataset.csv")

    it = items.merge(prods[["product_id", "product_category_name"]], on="product_id", how="left")
    agg = it.groupby("order_id").agg(n_itens=("order_item_id", "count"),
                                     valor_produtos=("price", "sum"),
                                     valor_frete=("freight_value", "sum"))
    cat = it.sort_values("order_item_id").drop_duplicates("order_id").set_index("order_id")["product_category_name"]
    pay_main = (pays.sort_values(["order_id", "payment_value"], ascending=[True, False])
                .drop_duplicates("order_id").set_index("order_id")["payment_type"])
    parcelas = pays.groupby("order_id")["payment_installments"].max()

    df = orders.merge(custs, on="customer_id", how="left")
    df = df.join(agg, on="order_id").join(cat.rename("categoria"), on="order_id")
    df = df.join(pay_main.rename("pagamento"), on="order_id").join(parcelas.rename("parcelas"), on="order_id")

    curto = lambda s: s.astype(str).str[:10]
    assert curto(df["order_id"]).is_unique
    data = lambda c: pd.to_datetime(df[c])

    base = pd.DataFrame({
        "pedido_id": curto(df["order_id"]),
        "cliente_id": curto(df["customer_unique_id"]),
        "data_compra": data("order_purchase_timestamp").dt.strftime("%Y-%m-%d %H:%M:%S"),
        "status": df["order_status"].map(STATUS),
        "cidade": df["customer_city"],
        "estado": df["customer_state"],
        "categoria": df["categoria"],
        "n_itens": df["n_itens"].fillna(0).astype(int),
        "valor_produtos": df["valor_produtos"].round(2),
        "valor_frete": df["valor_frete"].round(2),
        "valor_total": (df["valor_produtos"] + df["valor_frete"]).round(2),
        "pagamento": df["pagamento"].map(PAGAMENTO),
        "parcelas": df["parcelas"].astype("Int64"),
        "data_aprovacao": data("order_approved_at").dt.strftime("%Y-%m-%d %H:%M:%S"),
        "data_envio": data("order_delivered_carrier_date").dt.strftime("%Y-%m-%d %H:%M:%S"),
        "data_entrega": data("order_delivered_customer_date").dt.strftime("%Y-%m-%d"),
        "data_entrega_prevista": data("order_estimated_delivery_date").dt.strftime("%Y-%m-%d"),
    }).sort_values("data_compra").reset_index(drop=True)

    av = pd.DataFrame({
        "avaliacao_id": curto(revs["review_id"]),
        "pedido_id": curto(revs["order_id"]),
        "nota": revs["review_score"],
        "titulo": revs["review_comment_title"],
        "comentario": revs["review_comment_message"],
        "data_avaliacao": pd.to_datetime(revs["review_creation_date"]).dt.strftime("%Y-%m-%d"),
    })
    return base, av


# ------------------------------------------------------------------ sujeira plantada
def sujar(base, seed=SEMENTE):
    rng = np.random.default_rng(seed)
    d = base.copy()
    d["valor_total"] = d["valor_total"].astype(object)
    n = len(d)

    # 1) valores com vírgula decimal, como texto (exportação de uma planilha)
    idx = rng.choice(n, size=int(n * 0.04), replace=False)
    d.loc[idx, "valor_total"] = [f"{v:.2f}".replace(".", ",") if pd.notna(v) else v
                                 for v in d.loc[idx, "valor_total"]]
    # 2) estado digitado de formas diferentes
    idx = rng.choice(n, size=int(n * 0.02), replace=False)
    variantes = [lambda s: s.lower(), lambda s: f" {s}", lambda s: s.capitalize() + " "]
    d.loc[idx, "estado"] = [variantes[i % 3](s) for i, s in enumerate(d.loc[idx, "estado"])]
    # 3) pedidos de teste da engenharia
    modelo = d[d["status"] == "entregue"].iloc[[n // 5, n // 3, n // 2, (3 * n) // 4]].copy()
    modelo["pedido_id"] = ["teste00001", "teste00002", "teste00003", "teste00004"]
    modelo["cliente_id"] = "teste00000"
    modelo["cidade"] = "teste"
    modelo["estado"] = "SP"
    modelo["categoria"] = np.nan
    modelo["valor_produtos"] = 99999.99
    modelo["valor_frete"] = 0.0
    modelo["valor_total"] = 99999.99
    # 4) linhas duplicadas na exportação
    dup = d.iloc[rng.choice(n, size=int(n * 0.015), replace=False)]
    d = pd.concat([d, modelo, dup], ignore_index=True)
    d = d.iloc[rng.permutation(len(d))].sort_values("data_compra", kind="stable").reset_index(drop=True)
    return d, len(dup), len(modelo)


# ------------------------------------------------------------------ respostas de referência
def respostas(versao, sujo, limpo, av, n_dup, n_teste):
    MIN = MIN_PEDIDOS[versao]
    r = {}
    validos = ~limpo["status"].isin(["cancelado", "indisponivel"])

    # T1 — CEO
    r["T1_linhas"] = (len(sujo), 0)
    r["T1_inicio"] = (sujo["data_compra"].min()[:10], 0)
    r["T1_entregue"] = ((sujo["status"] == "entregue").mean() * 100, 1)
    r["T1_nulos"] = (sujo.isna().sum().idxmax(), 0)

    # T2 — Financeiro
    planilha = sujo["valor_total"].map(lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else np.nan).sum()
    r["T2_duplicadas"] = (n_dup, 0)
    r["T2_estados"] = (limpo["estado"].nunique(), 0)
    r["T2_teste"] = (n_teste, 0)
    r["T2_2018"] = ((pd.to_datetime(limpo["data_compra"]).dt.year == 2018).sum(), 0)
    r["T2_faturamento"] = (limpo.loc[validos, "valor_total"].sum(), 0)
    r["T2_planilha"] = (planilha, 0)

    # T3 — Marketing
    dc = pd.to_datetime(limpo["data_compra"])
    bf = limpo[dc.dt.strftime("%Y-%m-%d") == "2017-11-24"]
    sp = bf[bf["estado"] == "SP"]
    nov = limpo[(dc >= "2017-11-01") & (dc < "2017-11-24")]
    media_dia = len(nov) / 23
    r["T3_pedidos_bf"] = (len(bf), 0)
    r["T3_clientes_sp"] = (sp["cliente_id"].nunique(), 0)
    r["T3_ticket_sp"] = (sp["valor_total"].mean(), 2)
    r["T3_vezes"] = (len(bf) / media_dia, 1)

    # T4 — Logística
    ent = limpo[(limpo["status"] == "entregue") & limpo["data_entrega"].notna()].copy()
    ent["data_entrega"] = pd.to_datetime(ent["data_entrega"])
    ent["data_entrega_prevista"] = pd.to_datetime(ent["data_entrega_prevista"])
    ent["atrasado"] = ent["data_entrega"] > ent["data_entrega_prevista"]
    por_estado = ent.groupby("estado")["atrasado"].agg(["mean", "count"])
    por_estado = por_estado[por_estado["count"] >= MIN]
    ent["mes"] = pd.to_datetime(ent["data_compra"]).dt.strftime("%Y-%m")
    por_mes = ent.groupby("mes")["atrasado"].agg(["mean", "count"])
    por_mes = por_mes[por_mes["count"] >= MIN]
    atrasados = ent[ent["atrasado"]]
    r["T4_pct"] = (ent["atrasado"].mean() * 100, 1)
    r["T4_pior_estado"] = (por_estado["mean"].idxmax(), 0)
    r["T4_se"] = (ent.loc[ent["estado"] == "SE", "atrasado"].mean() * 100, 1)
    r["T4_dias"] = ((atrasados["data_entrega"] - atrasados["data_entrega_prevista"]).dt.days.mean(), 1)
    r["T4_mes"] = (por_mes["mean"].idxmax(), 0)

    # T5 — Produto
    m = limpo.merge(av, on="pedido_id", how="left")
    por_cat = m.groupby("categoria")["nota"].agg(["mean", "count"])
    por_cat = por_cat[por_cat["count"] >= MIN]
    contagem = av["pedido_id"].value_counts()
    r["T5_nota_media"] = (m["nota"].mean(), 2)
    r["T5_pior_categoria"] = (por_cat["mean"].idxmin(), 0)
    r["T5_linhas_extra"] = (len(m) - len(limpo), 0)
    r["T5_multiplas"] = ((contagem > 1).sum(), 0)

    # T6 — Diretoria
    e2 = ent.merge(av[["pedido_id", "nota"]], on="pedido_id", how="inner")
    medias = e2.groupby("atrasado")["nota"].mean()
    r["T6_diferenca"] = (medias[False] - medias[True], 2)
    return r


def main(pasta_olist, pasta_saida):
    out = Path(pasta_saida)
    out.mkdir(parents=True, exist_ok=True)
    base, av = montar_base(pasta_olist)
    rng = np.random.default_rng(SEMENTE)
    amostra = np.sort(rng.choice(len(base), size=TAMANHO_SHORT, replace=False))
    versoes = {"short": base.iloc[amostra].reset_index(drop=True), "full": base}

    for v, limpo in versoes.items():
        av_v = av[av["pedido_id"].isin(limpo["pedido_id"])].reset_index(drop=True)
        sujo, n_dup, n_teste = sujar(limpo)
        sujo.to_csv(out / f"pedidos_sujo_{v}.csv", index=False)
        limpo.to_csv(out / f"pedidos_limpo_{v}.csv", index=False)
        av_v.to_csv(out / f"avaliacoes_{v}.csv", index=False)

        # relê do CSV para as respostas refletirem exatamente o que os alunos vão carregar
        sujo_r = pd.read_csv(out / f"pedidos_sujo_{v}.csv")
        limpo_r = pd.read_csv(out / f"pedidos_limpo_{v}.csv")
        av_r = pd.read_csv(out / f"avaliacoes_{v}.csv")
        r = respostas(v, sujo_r, limpo_r, av_r, n_dup, n_teste)

        hashes, linhas = {}, [f"# Gabarito de códigos ({v}) — só para monitores\n",
                              "| Tíquete | Resposta | Código |", "|---|---|---|"]
        for k, (valor, casas) in r.items():
            t = normalizar(valor, casas)
            hashes[k] = {"hash": hash_resposta(t), "casas": casas}
            linhas.append(f"| {k} | {t} | {codigo(t)} |")
        painel = {"faturamento_painel": round(float(r["T2_faturamento"][0]), 2),
                  "soma_planilha": round(float(r["T2_planilha"][0]), 2)}
        json.dump({"versao": v, "min_pedidos": MIN_PEDIDOS[v], "painel": painel, "respostas": hashes},
                  open(out / f"respostas_{v}.json", "w"), indent=1)
        (out / f"codigos_{v}.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
        print(f"[{v}] sujo={len(sujo)} limpo={len(limpo)} avaliacoes={len(av_v)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
