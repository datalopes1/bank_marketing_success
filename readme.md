# Predição do Sucesso de Campanhas de Marketing - Bank Marketing Dataset

Os dados estão relacionados com campanhas de marketing direto de uma instituição bancária portuguesa. As campanhas de marketing foram baseadas em chamadas telefônicas. Muitas vezes, mais de um contato com o mesmo cliente era necessário para verificar se o produto (*term deposit*) seria ou não contratado. Os dados foram encontrados no [UC Irvine - Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) e disponibilizados por S. Moro, R. Laureano e P. Cortez. 

![](https://i.imgur.com/6a1BREy.jpeg)

### Features
|Coluna|Descrição|
|---|---|
|age|Idade|
|job|Profissão|
|marital|Estado civil|
|education|Nível de educação|
|default|Inadimplência (yes/no)|
|balance|Média de balanço anual|
|housing|Financiamento imobiliário (yes/no)|
|loan|Empréstimo pessoa (yes/no)|
|contact|Forma de contato|
|day|Dia do último contato no mês|
|month|Mês do último contato|
|duration|Duração das ligações em segundos|
|campaign|Número de contatos feitos durante esta campanha|
|pdays|Dias desde o contato em uma campanha anterior (-1 significa que não foi contatado)|
|previous|Número de contatos feitos em uma campanha anterior|
|poutcome|Resultado da última campanha|
|y|Sucesso dessa campanha (yes/no)|
## Metas e objetivos

Os objetivos desse projeto são (1) fazer uma análise exploratória de dados para buscar insights para futuras campanhas, e (2) criar um modelo de Machine Learning para predizer o sucesso das campanhas a partir dos dados nesse conjunto.

### Resultados
#### Conclusões pós-Análise
- Clientes mais velhos tem maior probabilidade de aceite;
- Com segundo maior número de clientes dentro do banco, pessoas em cargos de gerência tem maior sucetividade a aceitar o *term deposit*, seguidos por técnicos e prestadores de serviços. Campanhas segmentadas a categórias profissionais podem trazer um bom resultado.
- Clientes adimplentes e sem crédito comprometido com empréstimos imobiliários ou pessoais tem maior probabilidade de aceitar os produtos, é interessante focar nestes clientes nas campanhas;
- O meio de contato mais efetivo é o celular;
- O segundo e o terceiro trimestre são os períodos de maior sucesso nas campanhas, e podem ser trabalhadas campanhas com um orçamento maior nestes meses. Em especial Maio-Abril e Julho-Agosto
- A média de tempo das ligações com resultado positivo é de 7,25 minutos, essa pode ser uma boa métrica para os colaboradores do telemarketing assim como o número de ligações que deve ser entre 2 e 7;
- Clientes com respostas positivas em campanhas anteriores tem tendência de aceitar ofertas em novas campanhas. 

Portanto sugiro para as próximas campanhas:
1. Iniciar por (1) clientes os contatos por clientes adimplentes e com crédito livre, (2) clientes que aceitaram o produtos na última campanha;
2. Realizar entre de 2 a 7 ligações, e só voltar a ligar para clientes que passaram deste limite quando os contatos a todos forem concluídos dentro deste limite;
3. Ao conseguir o contato, buscar alongar a ligação até no máximo 7 minutos, ao passar desse prazo de tempo as probabilidades de aceite diminuem;
4. Criar campanhas segmentadas por profissões e tentar colocar elas no espaço de tempo com maior probabilidade de sucesso, como o périodo Julho-Agosto por exemplo. 

#### Modelo de Classificação
Utilizando o LGBMClassifier atingi as seguintes métricas:
|Métrica|Resultado|
|-|-|
|Log Loss|0.2030|
|Accuracy|0.9050|
|ROC AUC|0.9290|

## Ferramentas utilizadas
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
### Ferramentas utilizadas
#### Manipulação de dados
- Pandas, NumPy, Scipy
#### Visualização de Dados
- Seaborn, Matplotlib
#### Machine Learning
- LightGBM, Scikit-learn, feature_engine, category_encoders

# Análise Exploratória de Dados (EDA)
## Feature a feature
![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot1.png?raw=true)

Existe um assimetria na distribuição da idade causada pela quantidade de clientes na terceira idade.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot2.png?raw=true)

- O número de clientes acima dos 60 anos é 487, eles são 1.08% do total.
- A maioria dos clientes estão entre o meio de seus 30 anos e fim dos 40. Existe uma quantidade relavativamente baixa de idosos em relação ao restante.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot3.png?raw=true)

No inglês "Blue-collar" se refere a empregos mais braçais, geralmente ligados a construção civil e prestação de serviços como operadores de máquinas, caminhoneiros, eletricistas e etc. Estes profissionais mais de 20% dos clientes do banco, seguidos por pessoas em cargos de gerência e técnicos.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot5.png?raw=true)

Quase 70% dos clientes do banco não possuem uma graduação, o que se reflete na alta quantidade de prestadores de serviços e técnicos entre eles.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot6.png?raw=true)


![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot7.png?raw=true)

Como se tratam de dados bancários, é normal a existente de inviduos com saldos e balanços em valores muito maiores que a média.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot8.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot9.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot10.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot11.png?raw=true)

O volume de contatos se acumula entre o fim do segundo trimestre o começo do segundo, sendo maio o mês com maior quantidade de contatos.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot12.png?raw=true)

- A duração média das ligações é: 258.1630797814691
- A mediana da duração das ligações é: 180.0
- O desvio padrão da duração das ligações é: 257.52781226517095

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot13.png?raw=true)

Existe uma alta quantidade de outliers nas ligações, outra variável que precisa de uma limpeza prévia, uma ligação de 1000 segundos tem aproximadamente 16 minutos é um tempo longo de ligação. Em breve vamos ver a relação entre o tempo de ligação e o sucesso na assintura do term deposit.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot14.png?raw=true)

- Média de ligações durante a campanha: 2.763840658246887
- Mediana de ligações durante a campanha: 2.0
- Desvio padrão do número de ligações: 3.0980208832802205

Essa é outra variável que sofre com outliers, e é um ponto de alerta, um cliente que recebe mais de 50 ligações sobre uma campanha pode se tornar alguém em ponto de churn ou um processo legal contra instituição.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot15.png?raw=true)
A porcentagem de clientes que não foram contatados em campanhas anteriores: 81.74%

Acredito que a variável de número de contatos seja mais representativa que a de número de dias. A um volume alto de clientes não contatados, vamos ver se isso se confirma na variável de número de contatos.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot16.png?raw=true)
##### Considerando o número de contatos
A porcentagem de clientes que não foram contatados em campanhas anteriores: 81.74%

Vou preferir para o modelo de Machine Learning usar essa feature, e remover a anterior. Existe um alto volume de cliente não contatados, o que também mostra espaço de ação livre para oferta de produtos bancários.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot17.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot18.png?raw=true)

A campanha tem uma taxa de sucesso de somente 11%.

## Features x Target
![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot19.png?raw=true)

Existe uma tendência no aceite do *term deposit* em clientes mais velhos.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot20.png?raw=true)

Com segundo maior número de clientes dentro do banco, pessoas em cargos de gerência tem maior sucetividade a aceitar o *term deposit*, seguidos por técnicos e prestadores de serviços. É interessante buscar estes profissionais para oferta também de outros produtos.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot21.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot22.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot23.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot24.png?raw=true)

Os outliers dificultam um pouco a visualização, vamos realizar a remoção dos outliers para checar a distribuição.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot25.png?raw=true)

Assim podemos confirmar a baixa relação entre balanço anual, e o sucesso da campanha.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot26.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot27.png?raw=true)

O segundo e o terceiro trimestre são os períodos de maior sucesso nas campanhas, e podem ser trabalhadas campanhas com um orçamento maior nestes meses. Em especial Maio-Abril e Julho-Agosto.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot28.png?raw=true)

##### Ligações mais longas tem maior chance de sucesso, mas o quão longas?
- A duração média de uma ligação com sucesso: 7.25 minutos
- A duração mínima de uma ligação com sucesso: 0.13 minutos
- A duração mínima de uma ligação com sucesso: 17.17 minutos

A média de 7 minutos pode se tornar uma métrica para o treinamento dos colabodores de telemarketing.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot29.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot30.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot31.png?raw=true)

Novamente os outliers dificultam um pouco a interpretação, vamos observar sem eles.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot32.png?raw=true)

- Número médio de ligações para um resultado positivo: 2.065754465982516.
- Número mínimo de ligações para um resultado positivo: 1.
- Número máximo de ligações para um resultado positivo: 12.
- Desvio padrão do número de ligações: 1.5776938358631944

Apesar do número máximo de ligações ser de 12, podemos usar como métrica uma quantidade de até 3 desvios padrões da média (como em uma distribuição normal) tornando 7 o máximo de ligações feitas a um cliente, podendo assim otimizar tempo e recursos do time de telemarketing.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot33.png?raw=true)

Vou realizar uma limpeza de outliers para melhor visualização

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot34.png?raw=true)

Pessoas com maior número de contatos em campanhas anteriores tem maiores chances de aceitar o term deposit.

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot35.png?raw=true)

# Modelo de Classificação 
O modelo escolhido para este projeto foi LGBMClassifier da biblioteca LightGBM. Com ele atingi as seguintes métricas. 

### Métricas do Modelo
#### Treino
|Métrica|Resultado|
|-|-|
|Log Loss|0.1816|
|Accuracy|0.9176|
|ROC AUC|0.9487|

#### Teste
|Métrica|Resultado|
|-|-|
|Log Loss|0.2030|
|Accuracy|0.9050|
|ROC AUC|0.9290|

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot36.png?raw=true)

![](https://github.com/datalopes1/bank_marketing_success/blob/main/doc/img/plot37.png?raw=true)