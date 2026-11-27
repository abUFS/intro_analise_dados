"""
Tabelas extras da Balcão para o Dia 3 (rode depois de preparar_dados.py).

Uso:
    python preparar_dia3.py <pasta_com_csvs_da_olist> <pasta_de_saida>

Gera, para "short" e "full":
    itens_<v>.csv        um produto vendido por linha (preço, frete, categoria, peso, medidas, fotos, vendedor)
    vendedores_<v>.csv   cidade e estado de cada lojista
    pagamentos_<v>.csv   cada pagamento de um pedido (um pedido pode ter vários)

A versão short usa os mesmos pedidos de pedidos_limpo_short.csv.
Fonte: Olist (CC BY-NC-SA 4.0).
"""
import sys
from pathlib import Path

import pandas as pd

PAGAMENTO = {"credit_card": "cartao_credito", "boleto": "boleto", "voucher": "voucher",
             "debit_card": "cartao_debito", "not_defined": "nao_definido"}
curto = lambda s: s.astype(str).str[:10]


def main(pasta_olist, pasta_saida):
    p, out = Path(pasta_olist), Path(pasta_saida)
    items = pd.read_csv(p / "olist_order_items_dataset.csv")
    prods = pd.read_csv(p / "olist_products_dataset.csv")
    sellers = pd.read_csv(p / "olist_sellers_dataset.csv")
    pays = pd.read_csv(p / "olist_order_payments_dataset.csv")

    it = items.merge(prods, on="product_id", how="left")
    itens = pd.DataFrame({
        "pedido_id": curto(it["order_id"]),
        "item": it["order_item_id"],
        "produto_id": curto(it["product_id"]),
        "vendedor_id": curto(it["seller_id"]),
        "categoria": it["product_category_name"],
        "preco": it["price"].round(2),
        "frete": it["freight_value"].round(2),
        "peso_g": it["product_weight_g"],
        "comprimento_cm": it["product_length_cm"],
        "altura_cm": it["product_height_cm"],
        "largura_cm": it["product_width_cm"],
        "fotos": it["product_photos_qty"],
    })
    vendedores = pd.DataFrame({
        "vendedor_id": curto(sellers["seller_id"]),
        "cidade": sellers["seller_city"],
        "estado": sellers["seller_state"],
    })
    pagamentos = pd.DataFrame({
        "pedido_id": curto(pays["order_id"]),
        "sequencia": pays["payment_sequential"],
        "tipo": pays["payment_type"].map(PAGAMENTO),
        "parcelas": pays["payment_installments"],
        "valor": pays["payment_value"].round(2),
    }).sort_values(["pedido_id", "sequencia"])

    for v in ("short", "full"):
        ids = set(pd.read_csv(out / f"pedidos_limpo_{v}.csv", usecols=["pedido_id"])["pedido_id"])
        it_v = itens[itens["pedido_id"].isin(ids)].reset_index(drop=True)
        ve_v = vendedores[vendedores["vendedor_id"].isin(set(it_v["vendedor_id"]))].reset_index(drop=True)
        pg_v = pagamentos[pagamentos["pedido_id"].isin(ids)].reset_index(drop=True)
        it_v.to_csv(out / f"itens_{v}.csv", index=False)
        ve_v.to_csv(out / f"vendedores_{v}.csv", index=False)
        pg_v.to_csv(out / f"pagamentos_{v}.csv", index=False)
        print(f"[{v}] itens={len(it_v)} vendedores={len(ve_v)} pagamentos={len(pg_v)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
