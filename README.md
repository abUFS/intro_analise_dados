# Minicurso de Análise de Dados com Python · Monitoria Prog I 2026

Materiais do Dia 2 (empresa fictícia **Balcão**).

## Dados (`dados/`)

| Arquivo | O que é |
|---|---|
| `pedidos_sujo_{short,full}.csv` | Pedidos como "a engenharia exportou", com problemas plantados (tíquetes 1 e 2) |
| `pedidos_limpo_{short,full}.csv` | A mesma tabela já limpa (a partir do tíquete 3) |
| `avaliacoes_{short,full}.csv` | Avaliações dos clientes (tíquete 5) |
| `respostas_{short,full}.json` | Hashes usados pela função `conferir()` do notebook |

- **full**: todos os ~99 mil pedidos · **short**: amostra de 8 mil pedidos (internet lenta).
- Para regenerar: `python preparar_dados.py <pasta_com_csvs_da_olist> dados`

## Fonte e licença

Dados reais e anonimizados do **Brazilian E-Commerce Public Dataset by Olist**
(Kaggle: `olistbr/brazilian-ecommerce`), licença **CC BY-NC-SA 4.0**.
Os arquivos derivados deste repositório seguem a mesma licença.
A empresa "Balcão" e as pessoas citadas nos tíquetes são fictícias.
