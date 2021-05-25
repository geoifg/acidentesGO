
# Script: Análise e modelagem dos acidentes de trânsito nas rodovias federais no estado de Goiás em 2017 e 2018 ------------------------------------------------------------
# Autor: Edipo Henrique Cremon (edipo.cremon@ifg.edu.br)
# Data de criacao: 2021-03-02
# R version 4.0.2 (2020-06-22)
#

# Defina seu diretorio de trabalho. Ex: "D:/MESTRADO"
setwd("D:/MESTRADO/")


##Aqui iremos instalar os pacores que iremos utilizar
install.packages(c('randomForest','caret', 'rpart', 'rpart.plot', 'dplyr'),dependencies=TRUE)

##Apos instalar os pacores, e necessario executa-los
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)

#Lendo o arquivo csv das ocorrencias de acidentes de 2017 e 2018, para esse dado daremos o nome de "dados_acid"
dados_acid <- read.csv("dados_all_numerico.csv", 
                       header = TRUE, sep = ";", quote = "\"",
                       dec = ".")

#Conferindo a estatastica basica sobre cada variavel (coluna) do data.drame
summary(dados_acid)


#conferindo o tipo de variavel (numerica, texto, etc) do data.frame
str(dados_acid)

#Excluindo colunas que nao serao usadas
dados_acid$FID <- NULL
dados_acid$tipo_acide <- NULL

#Conferindo os primeiros dados
head(dados_acid)

#Converter as variaveis de numericas para factors (categoricas)
dados_acid$dia_semana <- as.factor(dados_acid$dia_semana)
dados_acid$fase_dia <- as.factor(dados_acid$fase_dia)
dados_acid$sentido_vi <- as.factor(dados_acid$sentido_vi)
dados_acid$condicao_m <- as.factor(dados_acid$condicao_m)
dados_acid$vl_br <- as.factor(dados_acid$vl_br)
dados_acid$nm_tipo_tr <- as.factor(dados_acid$nm_tipo_tr)
dados_acid$ds_superfi <- as.factor(dados_acid$ds_superfi)
dados_acid$ds_legenda <- as.factor(dados_acid$ds_legenda)
dados_acid$TIPO <- as.factor(dados_acid$TIPO)
dados_acid$causa_acid <- as.factor(dados_acid$causa_acid)
dados_acid$Acidente <- as.factor(dados_acid$Acidente)

#Conferir os dados convertidos para factors
str(dados_acid)

##Verificar quantas amostras ha
nrow(dados_acid)

#Dividir as amostras de treinamento e valida??o
#No caso usaremos as mesmas amostras para treinamento e validacaoo, pois usaremos a abordagem
#de validacaoo-cruzada de k-fold, onde as todo conjunto amostral e dividido em 5 partes
#Sao usadas 4 partes para gerar o modelo e 1 para validacao. Isso e repetido com todas as partes.
#O melhor modelo e escolhido.
set.seed(100)
dados_acid2 <- as.vector(createDataPartition(dados_acid$Acidente, list=FALSE, p=0.7))
treinamento_acid <- dados_acid[dados_acid2,]
validacao_acid <- dados_acid[-dados_acid2,]

##Verificar quantas amostras de treinamento
nrow(treinamento_acid)

##Verificar quantas amostras de validao
nrow(validacao_acid)

levels(validacao_acid$Acidente) <- make.names(levels(factor(validacao_acid$Acidente)))

table(validacao_acid$Acidente)

#Definir as variaveis preditoras e a resposta
preditoras <- c("data_conse", "dia_semana", "horario", "fase_dia",
                "sentido_vi", "condicao_m", "vl_br", "nm_tipo_tr",
                "vl_km_inic", "vl_km_fina",
                "vl_extensa", "ds_superfi", "ds_legenda", "Dist_km", "TIPO",
                "S_med", "S_sd", "CV_med",  "CV_sd", "P_AREA", "P_PERIM", "P_FRACDIM",
                "P_PERARAT", "P_COMPAC","PBOX_AREA",  "PBOX_PERIM", "PBOX_LEN",
                "PBOX_WIDTH", "POL_ANGLE", "PELLIP_FIT", "PGYRATIUS", "POLRADIUS",
                "PCIRCLE", "PSAHPEIDX", "PDENSITY", "PRECTFIT")

#O comando trainContol determinara a abordagem do classificador
#o comando classProbs e summaryFunction so funciona com variaveis do tipo texto
control <- trainControl(method = "cv",
                        number = 5,
                        classProbs = TRUE,
                        savePredictions = TRUE,
                        summaryFunction = twoClassSummary)

levels(treinamento_acid$Acidente) <- make.names(levels(factor(treinamento_acid$Acidente)))

table(treinamento_acid$Acidente)

##Elaborando o modelo RF
set.seed(100)
rf_acidente <- train(x = treinamento_acid[, preditoras], y = treinamento_acid$Acidente, method = "rf",
                     trControl=control, metric = 'ROC')

print(rf_acidente)

#Matriz de confusao com os dados independentes de validacao
valid_rf <- predict(rf_acidente, validacao_acid)
RF_table <- table(validacao_acid$Acidente, valid_rf)

confusionMatrix(RF_table)

##Elaborando o modelo C4.5
set.seed(100)
model_c45 <- train(x = treinamento_acid[, preditoras], y = treinamento_acid$Acidente, method = "J48",
                   trControl=control, metric = 'ROC')

print(model_c45)


#Matriz de confusao C4.5 com os dados independentes de validacao
valid_c45 <- predict(model_c45, validacao_acid)
c45_table <- table(validacao_acid$Acidente, valid_c45)

confusionMatrix(c45_table)

##Elaborando o modelo C5.0
library(C50)

grid <- expand.grid(.model = "tree",.trials = c(1:50),.winnow = c(TRUE, FALSE))

set.seed(100)
c50_acidente <- train(x = treinamento_acid[, preditoras], y = treinamento_acid$Acidente, method = "C5.0",
                      trControl=control, metric = 'ROC', tuneGrid = grid)

print(c50_acidente)

#Matriz de confusao C5.0 com os dados independentes de validacao
valid_c50 <- predict(c50_acidente, validacao_acid)
c50_table <- table(validacao_acid$Acidente, valid_c50)

confusionMatrix(c50_table)


##Elaborando o modelo CART


set.seed(100)
model_CART <- train(x = treinamento_acid[, preditoras], y = treinamento_acid$Acidente, method = "rpart",
                    trControl=control, metric = 'ROC')

print(model_CART)


#Matriz de confusao CART com os dados independentes de validacao
valid_CART <- predict(model_CART, validacao_acid)
CART_table <- table(validacao_acid$Acidente, valid_CART)

confusionMatrix(CART_table)


##Plot comparando os modelos gerais para os acidentes
comparacao <- resamples(list(CART=model_CART, C4.5=model_c45, C5.0=c50_acidente, RF=rf_acidente))
# boxplots das validacoes
bwplot(comparacao)

summary(comparacao)

#Obtendo a import?ncia das variaveis
#Para a opcao varImp  type = 2 (default) e utilizada a media do decrescimo medio de Gini 'MeanDecreaseGini' 
# que se baseia no indice de impureza de Gini utilizado para o calculo dos nos das arvores de decisao. 
#Alternativamente, voce pode definir o tipo = 1, entao a medida calculada e a diminuicao media da exatidao (mean decrease in accuracy) .
#Algoritmo RF
plot(varImp(rf_acidente,type=2))
importancia <- varImp(rf_acidente)

plot(importancia, top=15, xlim=c(-5, 105))

#Outra opcao de grafico
dotPlot(importancia)

#Algoritmo C4.5
plot(varImp(model_c45,type=2))
importancia_c45 <- varImp(model_c45)

plot(importancia_c45, top=15, xlim=c(-5, 105))

#Algoritmo C5.0
plot(varImp(c50_acidente,type=2))
importancia_c50 <- varImp(c50_acidente)

plot(importancia_c50, top=15, xlim=c(-5, 105) )

#Algoritmo CART
plot(varImp(model_CART))
importancia_CART <- varImp(model_CART)

plot(importancia_CART, top=15, xlim=c(-5, 105))


##Calcular ROC e AUC
#Matriz de confusao com os dados independentes de validacao
valid_rf_causa <- predict(rf_causa, validacao_acid)
RF_table_causa <- table(validacao_acid$causa_acid, valid_rf_causa)

confusionMatrix(RF_table_causa)
## Obtendo a curva ROC a partir dos dados de validacao
library(ROCR)
library(pROC) 
roc_rf <- roc(validacao_acid$Acidente,
           predict(rf_acidente, validacao_acid, type = "prob")[,1],
           levels = rev(levels(validacao_acid$Acidente)))
roc_rf


roc_c50 <- roc(validacao_acid$Acidente,
              predict(c50_acidente, validacao_acid, type = "prob")[,1],
              levels = rev(levels(validacao_acid$Acidente)))
roc_c50

roc_c45 <- roc(validacao_acid$Acidente,
               predict(model_c45, validacao_acid, type = "prob")[,1],
               levels = rev(levels(validacao_acid$Acidente)))
roc_c45

roc_cart <- roc(validacao_acid$Acidente,
               predict(model_CART, validacao_acid, type = "prob")[,1],
               levels = rev(levels(validacao_acid$Acidente)))
roc_cart



#Obtendo o valor de AUC
auc(roc_rf)

#Obtendo o plot da curva ROC
plot(roc)

roc_rose <- plot(roc_rf, xlim=c(1,0), ylim=c(0,1), print.auc = TRUE, col = "green", print.auc.y = .5)
roc_rose <- plot(roc_c50, print.auc = TRUE, col = "blue", print.auc.y = .4, add = TRUE)
roc_rose <- plot(roc_cart, print.auc = TRUE, col = "red", print.auc.y = .2, add = TRUE)
roc_rose <- plot(roc_c45, print.auc = TRUE, col = "orange", print.auc.y = .3, add = TRUE)

