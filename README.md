# Brazilian Ecommerce Data Mining

Pipeline em Python que integra as tabelas do dataset público **Olist (Brazilian E-Commerce)** e aplica, em um único fluxo reproduzível, técnicas de mineração de dados: consolidação de tabelas, regras de associação, classificação de sentimento e árvore de decisão.

## Problema resolvido

Reúne diferentes técnicas de mineração e modelagem sobre dados reais de marketplace em um pipeline único, servindo como demonstração prática de análise de e-commerce de ponta a ponta.

## O que o pipeline entrega

- consolidação de 8 tabelas do Olist em uma base única;
- perfilamento estatístico da base integrada;
- regras de associação entre produtos (algoritmo Apriori);
- classificação simples de sentimento em comentários (Naive Bayes);
- árvore de decisão com exportação visual.

## Stack

- Python 3.10+
- pandas
- scikit-learn
- mlxtend
- matplotlib

## Estrutura

```text
brazilian-ecommerce-data-mining/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── external/        # CSVs do Olist (não versionados)
├── outputs/
│   └── figures/         # gráficos gerados
└── src/
    └── brazilian_ecommerce_data_mining.py
```

## Dados

O projeto usa o **Brazilian E-Commerce Public Dataset by Olist**, disponível no Kaggle:
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

Baixe os CSVs e coloque-os em `data/external/olist/`. São esperados os arquivos:

- `olist_customers_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`

Os dados não são versionados (ver `.gitignore`); cada máquina baixa a própria cópia.

## Como executar

```bash
# 1. crie e ative um ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. instale as dependências
pip install -r requirements.txt

# 3. rode o pipeline apontando para a pasta dos CSVs (a partir da raiz do projeto)
python src/brazilian_ecommerce_data_mining.py --data-dir data/external/olist
```

Parâmetro opcional `--sample-frac` controla a fração usada nas regras de associação (padrão `0.05`):

```bash
python src/brazilian_ecommerce_data_mining.py --data-dir data/external/olist --sample-frac 0.1
```

## Resultado esperado

Arquivos gerados em `outputs/`:

- `consolidated_dataset.csv` — base integrada
- `dataset_profile.csv` — perfil estatístico
- `association_rules.csv` — regras de associação
- `sentiment_predictions.csv` — previsões de sentimento
- `decision_tree_score.txt` — R² da árvore de decisão
- `figures/decision_tree.png` — visualização da árvore
