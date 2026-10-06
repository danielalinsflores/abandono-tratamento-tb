# 🏥 Previsão de Abandono de Tratamento de Tuberculose — Rio de Janeiro

Modelo de machine learning que prevê, a partir do perfil inicial de um paciente diagnosticado com tuberculose, a probabilidade dele abandonar o tratamento antes de completá-lo — permitindo priorização de busca ativa e intervenção precoce.

**Status:** ✅ Modelo treinado e validado

---

## 🎯 Contexto e motivação

O Brasil tem como meta nacional uma taxa de abandono de tratamento de TB **abaixo de 5%** (Ministério da Saúde). Abandono de tratamento é um dos maiores riscos de saúde pública em TB: favorece recidiva, resistência medicamentosa (TB-DR) e manutenção da cadeia de transmissão.

Diferente de um evento agudo (como dengue), abandono é um processo gradual — e por isso difícil de perceber a tempo. Este projeto usa o histórico de casos do SINAN-TB para identificar, logo no início do tratamento, quais pacientes têm maior risco de abandonar, dando à equipe de saúde uma janela de tempo real para agir.

---

## 📊 Principais achados

- **Taxa de abandono no RJ (2022-2024): 19,4%** — quase 4x acima da meta nacional de 5%, um indicador real de vulnerabilidade social na região.
- **Fatores de maior peso preditivo** (via importância de variáveis, Random Forest): uso de drogas, idade, histórico de abandono prévio, situação de rua e tipo de entrada no tratamento — consistente com a literatura epidemiológica sobre determinantes sociais do abandono de TB.
- **Modelo final: Regressão Logística** (não Random Forest), escolhida por recall superior (63% vs. 57%) e maior interpretabilidade — priorizando não deixar passar casos reais de risco em vez de maximizar acurácia geral.
- **Threshold de decisão ajustado para 0,4** (em vez do padrão 0,5): eleva o recall para 79%, equilíbrio prático entre capturar mais casos de risco e manter a lista de priorização operacionalmente viável para equipes de busca ativa.

---

## 🗂️ Fonte de dados

- **SINAN-TB**, via [PySUS](https://github.com/AlertaDengue/PySUS) (biblioteca Python para acesso ao FTP legado do DATASUS — Sinan Net).
- Recorte: Rio de Janeiro, 2022–2024.

---

## 🏗️ Arquitetura

abandono-tratamento-tb/
├── data/
│ ├── raw/ # parquet bruto baixado via PySUS
│ └── processed/ # dados limpos e com features
├── src/
│ ├── ingestion/ # download via PySUS
│ ├── cleaning/ # filtro RJ, criação do target
│ ├── features/ # engenharia de atributos
│ └── modeling/ # treino, avaliação, comparação de modelos
├── models/ # modelo treinado (.pkl) + threshold salvo
├── dashboards/
└── README.md


**Fluxo:** Ingestão (PySUS) → Limpeza (filtro RJ + target binário) → Engenharia de atributos (decodificação, one-hot encoding) → Split treino/teste estratificado → Treino e comparação de modelos → Ajuste de threshold → Persistência do modelo final.

---

## 🛠️ Stack técnica

- **Python** — pandas, scikit-learn, PySUS, joblib
- Regressão Logística e Random Forest (comparação)

---

## 🔧 Desafios técnicos e decisões de engenharia

### 1. Fonte de dados sem download em massa direto
**Problema:** diferente da dengue (Portal de Dados Abertos moderno), o SINAN-TB só está disponível via FTP legado do DATASUS.
**Solução:** uso da biblioteca PySUS, especializada em acessar essa infraestrutura legada.

### 2. Bug conhecido no catálogo da PySUS
**Problema:** consultas filtradas por doença (`disease="TUBE"`) retornavam zero arquivos, mesmo os dados existindo no servidor.
**Investigação:** identificado como um bug documentado publicamente — alguns arquivos têm um campo interno (`group_id`) nulo no catálogo, excluindo-os de buscas filtradas por grupo.
**Solução:** consulta sem filtro de grupo, com filtragem manual pelo nome do arquivo.

### 3. Tipo de dado inesperado no filtro de UF
**Problema:** filtro por `SG_UF == 33` não retornava nenhuma linha do RJ, mesmo os dados existindo.
**Investigação:** a coluna vem como texto (`"33"`), não número inteiro.
**Solução:** ajuste da comparação para string.

### 4. Definição do target de abandono
**Problema:** o campo de desfecho (`SITUA_ENCE`) tem 10 categorias possíveis, sem documentação explícita na massa de dados sobre qual(is) representa(m) abandono.
**Solução:** confirmação via dicionário oficial de dados do SINAN Net (Ministério da Saúde): target = `SITUA_ENCE` em [2 (Abandono), 10 (Abandono Primário)].

### 5. Vazamento de informação (avaliado e descartado)
**Decisão:** a variável `TRATAMENTO` (tipo de entrada) inclui a categoria "Reingresso após Abandono", que poderia parecer vazamento. Avaliado como **não sendo vazamento**: a informação reflete histórico anterior, disponível no momento do diagnóstico atual — equivalente a usar histórico de inadimplência em um modelo de crédito.

---

## 🤖 Modelagem

| Modelo | Recall (Abandono) | Precision (Abandono) | AUC-ROC |
|---|---|---|---|
| Regressão Logística (threshold 0,5) | 63% | 36% | 0,746 |
| Regressão Logística (threshold 0,4) | **79%** | 29% | 0,746 |
| Random Forest | 57% | 39% | 0,754 |

**Modelo final:** Regressão Logística com `class_weight="balanced"` e threshold ajustado para 0,4. Escolhida por recall superior e maior interpretabilidade, mesmo com AUC levemente inferior ao Random Forest — decisão de negócio: o custo de um falso negativo (abandono não detectado) é maior que o de um falso positivo (visita extra de busca ativa).

---

## ▶️ Como rodar

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/ingestion/download_data.py
python src/cleaning/clean_data.py
python src/features/build_features.py
python src/modeling/train_model.py


---

## 🗺️ Próximos passos

- [ ] Testar SMOTE como alternativa ao `class_weight="balanced"`
- [ ] Dashboard executivo com perfil de risco por região do RJ
- [ ] Validar achados contra literatura/dados oficiais de abandono de TB

---
