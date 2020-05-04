################################################################################
#Data Mining Assignment 
#Author: PRANAV SAXENA (119220273)
################################################################################

################################################################################
#Steps to run the code
################################################################################
#Set the current working directory to the NEWSGROUP folder which has only those 
#articles.This will enable the code to download only those 400 files.
#Run the code from top to bottom, except the last one "action = user_input()"
#Once you run action = user_input(), it will ask for various options related 
#to the project. You can choose from the option and can wait for the results.
#The progress will be displayed in the console.
################################################################################

rm(list = ls())
library(readr)
library(stringr)
library(hash)
library(mlr)
library(class)
library(randomForest)
library(tm)
library(tree)
library(e1071)
library(tokenizers)
library(caret)
library(glmnet)
library(stats)
library(mlr)
library(rpart)
library(e1071)

################################################################################
#Use tokenization and compute the top 200 most popular words (total occurrences) 
################################################################################

#Read all the files and create matrix of words with their count
read_files1 = function() {
  
  if (file.exists("data_train.txt")){
    file.remove("data_train.txt")
  }
  if (file.exists("data_test.txt")){
    file.remove("data_test.txt")
  }
  if (file.exists("wordstrain.csv")){
    file.remove("wordstrain.csv")
  }
  if (file.exists("wordstest.csv")){
    file.remove("wordstest.csv")
  }
  file_list = list.files(recursive = T)
  hash_words = hash()
  words_count = hash()
  unique_word = c()
  unique_count = c()
  j = 1
  hash_fname = hash()
  count=1
  
  for (file in file_list){
    
    lines = read_file(file)
    dat=str_replace_all(lines, "[\r\n]" , " ")
    dat=str_replace_all(dat,'"',' ')
    dat=str_replace_all(dat, '[^a-zA-Z]',' ')
    dat = tolower(dat)
    dat = strsplit(dat," ")[[1]]
    for(word in 1:length(dat)){
      if(dat[word] == "" ) {
        next
      }
      if(dat[word] %in% unique_word){
        index = which(unique_word == dat[word])
        unique_count[index] = unique_count[index] + 1
      }
      else{
        unique_word[j] =  dat[word]
        unique_count[j] = 1
        j = j+1
      }
    }
    hash_fname[count] = file
    count=count + 1
  }
  return(list(words=unique_word,uni_cnt=unique_count,fname=hash_fname))
}

#Display mostpopular 200 words(top 200 words which has highest frequency)
mostpopular200 = function(){
  
  print('Top 200 words matrix creation is in progress...')
  word_cnt = read_files1()
  uniq_words = word_cnt$words
  uniq_cnt = word_cnt$uni_cnt
  c=cbind(uniq_words,uniq_cnt)
  d = as.data.frame(c)
  d$uniq_cnt = as.numeric(uniq_cnt)
  sort_d = d[order(d$uniq_cnt,decreasing = T),]
  print(head(sort_d,n=200))  
}

######################################################
#Repeat the previous step, but now filter tokens by 
#length (min. size = 4 and max. size = 20). Please 
#indicate the total occurrences of the words. 
#####################################################

#Display mostpopular 200 wordswith specific range(top 200 words 
#which has highest frequency and specific length)
words_rnge = function(){
  
  print('Top 200 words matrix creation is in progress...')
  word_cnt = read_files1()
  word = word_cnt$words
  word_c = word_cnt$uni_cnt
  c = cbind(word,word_c)
  d = as.data.frame(c)
  upperFreq = 4
  characterLength = 20
  count = 0
  d$word = as.character(d$word)
  d$word_c = as.numeric(word_c)
  sort_d = d[order(d$word_c,decreasing = T),]
  df = as.data.frame(matrix(0,ncol=2,nrow=200))
  names(df) = c('words', 'count')
  for(i in 1:length(sort_d$word)){
    word1 = strsplit(sort_d$word[i],'')[[1]]
    if(length(word1)>= upperFreq && length(word1)<=characterLength ){
      if(count<=200){
        df[count,] = c(sort_d$word[i],sort_d$word_c[i])
        count = count + 1
      }
    }
  }
  print(df)
}

#Read the files and split them into test and train
#Return the train and test files
read_data = function(){ 
  
  if (file.exists("data_train.txt")){
    file.remove("data_train.txt")
  }
  if (file.exists("data_test.txt")){
    file.remove("data_test.txt")
  }
  if (file.exists("wordstrain.csv")){
    file.remove("wordstrain.csv")
  }
  if (file.exists("wordstest.csv")){
    file.remove("wordstest.csv")
  }
  file_list = list.files(recursive = T)
  set.seed(6405)
  i.index = sample(1:length(file_list),round(0.7*length(file_list)))
  file_train = file_list[i.index]
  file_test = file_list[-i.index]
  
  return(list(dat.train=file_train,dat.test=file_test))
}

#Read each file and create bag of words matrix for training and tetsing set
#Write it to csv file
eval_count = function(dat,action,dataset) { 
  
  if(dataset == 'train'){
    print(" Bag of Words creation for training data is in progress...")
  }
  else{
    print(" Bag of Words creation for testing data is in progress...")
  }
  for (fname in dat){
    lines = read_file(fname)
    dat=str_replace_all(lines, "[\r\n]" , " ")
    dat=str_replace_all(dat,'"',' ')
    
    if(action == 'robust'){
      dat=str_replace_all(dat, '[^a-zA-Z]',' ')
      dat = stripWhitespace(dat)
      dat = tolower(dat)
      dat = removeWords(dat, stopwords("en"))
      dat=textstem::lemmatize_strings(dat)
    }
    
    tb=table(tokenize_words(dat))
    tb.df=as.data.frame(tb)
    tb.df$class_y = fname
    
    if(dataset=='train'){
      write.table(tb.df, file='data_train.txt',row.names = F,col.names=F,append = T )
    }  
    else{
      write.table(tb.df, file='data_test.txt',row.names = F,col.names=F,append = T )
    }
  }
  
  if(dataset == 'train'){
    df = read.table('data_train.txt',header = F)
  }
  else{
    df = read.table('data_test.txt',header = F)
  }  
  
  lev = unique(df$V3)
  df1 = as.data.frame(matrix(0,ncol=(length(unique(df$V1)) ),nrow=length(lev)))
  names(df1) = unique(df$V1)
  k=1
  
  for( i in 1:nrow(df)){
    if(df$V3[i]!=lev[k]){
      k=k+1
    }
    df1[k,which(names(df1)==df$V1[i])] = df$V2[i]
  }
  
  df1$class_y=lev
  y=df1$class_y
  ychar = as.character(y)
  
  for ( i in 1:length(ychar)){
    if(str_detect(ychar[i],pattern = 'politics.guns')){
      ychar[i] = 'politics.guns'
    }
    else if(str_detect(ychar[i],pattern = 'politics.misc')){
      ychar[i] = 'politics.misc'
    }
    else if(str_detect(ychar[i],pattern = 'hardware')){
      ychar[i] = 'hardware'
    }
    else if(str_detect(ychar[i],pattern = 'electronics')){
      ychar[i] = 'electronics'
    }
  }
  
  df1$class_y = as.factor(ychar)
  
  if(dataset == 'train'){
    write.csv(df1,file = 'wordstrain.csv',row.names = F)
  }
  else{
    write.csv(df1,file = 'wordstest.csv',row.names = F)
  }
  if(dataset == 'train'){
    print(" Bag of Words creation for training set is completed and is written to wordstrain.csv")
  }
  else{
    print(" Bag of Words creation for testing set is completed and is written to wordstest.csv")
  }
}

#Read the CSV file having bag of words and return the training and test sets
read_data_csv = function(){
  
  data.train = read.csv('wordstrain.csv',header = T)
  data.test  = read.csv('wordstest.csv',header = T)
  
  x.train = data.train[,1:(length(data.train)-1)]
  y.train = data.train$class_y
  x.test = data.test[,1:(length(data.test)-1)]
  y.test = data.test$class_y
  names_test = names(x.test)  
  
  x.tr = x.train[,which(names(x.train)%in%names_test)]
  x.ts = x.test[,which(!names_test%in%names(x.train))]
  names_test = names(x.ts)
  x.tr[names_test] = 0
  train = as.data.frame(cbind(x.tr,y.train))
  names(train)[ncol(train)] = 'class_y'
  
  return(list(dat.train=train,dat.test=data.test))
}

#To check the class of the test data set and by applying Naive Bayes algorithm
check_class = function(x_test_words,df){
  
  p_hard=c()
  p_elec=c()
  p_polgun=c()
  p_polmisc=c()
  data=df
  
  for(i in 1:length(x_test_words)){
    k=1
    h=data[k,which(names(data)==x_test_words[i])]
    if(!is.numeric(h)){
      h=0
    }
    p_hard[x_test_words[i]] = log((h + 1) / (sum(data[k,ncol(data)-1])+ncol(data)))
    
    k=k+1
    el=data[k,which(names(data)==x_test_words[i])]
    if(!is.numeric(el)){
      el=0
    }
    p_elec[x_test_words[i]] = log((el + 1) / (sum(data[k,ncol(data)-1])+ncol(data)))
    
    k=k+1
    pg=data[k,which(names(data)==x_test_words[i])]
    if(!is.numeric(pg)){
      pg=0
    }
    p_polgun[x_test_words[i]] = log((pg + 1) / (sum(data[k,ncol(data)-1])+ncol(data)))
    
    k=k+1
    pm=data[k,which(names(data)==x_test_words[i])]
    if(!is.numeric(pm)){
      pm=0
    }
    p_polmisc[x_test_words[i]] = log((pm + 1) / (sum(data[k,ncol(data)-1])+ncol(data)))
  }
  
  max_p = sum(p_hard)
  class_pred = 'hardware'
  
  if(sum(p_elec) > max_p){
    max_p = sum(p_elec)
    class_pred = 'electronics'
  }
  if(sum(p_polgun) > max_p){
    max_p = sum(p_polgun)
    class_pred='politics.guns'
  }
  if(sum(p_polmisc) > max_p){
    max_p = sum(p_polmisc)
    class_pred = 'politics.misc'
  }
  
  return(class_pred)
}

#Formatting the training data to have counts of each words in particular class
format_data = function(x.train,y.train){
  
  df = as.data.frame(matrix(0,ncol=ncol(x.train),nrow=4))
  names(df) = names(x.train)
  df$classification=c('hardware','electronics','politics.guns','politics.misc')
  
  idx_hardware = which(y.train=='hardware')
  idx_elec = which(y.train=='electronics')
  idx_polgun = which(y.train=='politics.guns')
  idx_polmisc = which(y.train=='politics.misc')    
  
  for(i in 1:(ncol(x.train))){
    df[1,i] = sum(x.train[idx_hardware,i])
    df[2,i] = sum(x.train[idx_elec,i])
    df[3,i] = sum(x.train[idx_polgun,i])
    df[4,i] = sum(x.train[idx_polmisc,i])
  }
  
  return(df)
}

# Applying Naive Bayes algorithm
naive_bayes = function(data.train,data.test,action){
  
  print(" Naive Bayes is running...")
  x.train = data.train[,1:(length(data.train)-1)]
  y.train = data.train$class_y
  x.test = data.test[,1:(length(data.test)-1)]
  y.test = data.test$class_y
  n = nrow(x.train)
  
  if(action =='robust'){
    K = 3
    set.seed(6405)
    folds = cut(1:n, K, labels=FALSE)
    acc = numeric(K)
    recal = numeric(K)
    prcsn = numeric(K)
    fscore = numeric(K)
    
    for(k in 1:K){
      # training sample
      i = which(folds==k)
      x.tr = x.train[-i,]
      y.tr = y.train[-i]
      # validation sample
      x.val = x.train[i,]
      y.val = y.train[i]
      # train model on training data and validation:
      train.df = format_data(x.tr,y.tr)
      class_pred=c()
      for(j in 1:nrow(x.val)){
        x_val_words = names(x.val)[which(x.val[j,]>0)]
        class_pred[j] = check_class(x_val_words,train.df)
      }
      
      cm_nb_val = table(y.val,class_pred)
      per_matrix = performance_eval(cm_nb_val)
      acc[k] = per_matrix$acc
      recal[k] = per_matrix$recal
      prcsn[k] = per_matrix$precsn
      fscore[k] = per_matrix$fscore
    }
    
    train.df = format_data(x.train,y.train)
    class_pred=c()
    for(i in 1:nrow(x.test)){
      x_test_words = names(x.test)[which(x.test[i,]>0)]
      class_pred[i] = check_class(x_test_words,train.df)
    }
    
    cm_nb_test = table(y.test,class_pred)
    per_matrix_test = performance_eval(cm_nb_test)
    cat('\n')
    print(" Naive Bayes is completed...")
    
    return(list(acc_val = round(mean(acc),4),
                recal_val = round(mean(recal),4),
                precsn_val = round(mean(prcsn),4),
                fscore_val = round(mean(fscore),4),
                acc_test = per_matrix_test$acc,
                recal_test = per_matrix_test$recal,
                precsn_test = per_matrix_test$precsn,
                fscore_test = per_matrix_test$fscore,
                cm_val = cm_nb_val,
                cm_test = cm_nb_test
    ))
  }
  else{
    train.df = format_data(x.train,y.train)
    class_pred=c()
    for(i in 1:nrow(x.test)){
      x_test_words = names(x.test)[which(x.test[i,]>0)]
      class_pred[i] = check_class(x_test_words,train.df)
    }
    
    cm.nb=table(class_pred,y.test)
    per_matrix=performance_eval(cm.nb)
    print(" Naive Bayes is completed...")
    cat('\n')
    
    return(list(acc=per_matrix$acc,recal=per_matrix$recal,
                precsn=per_matrix$precsn,fscore=per_matrix$fscore,
                cm =cm.nb ))
  }
}

# Applying K-Nearest neighbour algorithm
knn1 = function(data.train,data.test,action){
  
  print(" KNN is running...")
  x.train = data.train[,1:(length(data.train)-1)]
  y.train = data.train$class_y
  x.test = data.test[,1:(length(data.test)-1)]
  y.test = data.test$class_y
  n = nrow(x.train)
  
  if(action == 'robust'){
    K=hyperparameter_tune('knn',data.train) 
    j = 3
    set.seed(6405)
    folds = cut(1:n, j, labels=FALSE)
    acc = numeric(j)
    recal = numeric(j)
    prcsn = numeric(j)
    fscore = numeric(j)
    
    for(k in 1:j){
      # training sample
      i = which(folds==k)
      x.tr = x.train[-i,]
      y.tr = y.train[-i]
      # validation sample
      x.val = x.train[i,]
      y.val = y.train[i]
      # train model on training data and validation:
      ko = knn(x.tr,x.val,y.tr, K)
      cm_knn_val = table(y.val,ko)
      per_matrix = performance_eval(cm_knn_val)
      acc[k] = per_matrix$acc
      recal[k] = per_matrix$recal
      prcsn[k] = per_matrix$precsn
      fscore[k] = per_matrix$fscore
    }
    
    # training the model on training and test data
    ko = knn(x.train,x.test,y.train, K)
    cm_knn_test = table(y.test,ko)
    per_matrix_test = performance_eval(cm_knn_test)
    cat('\n')
    print(" KNN is completed...")
    
    return(list(acc_val = round(mean(acc),4),
                recal_val = round(mean(recal),4),
                precsn_val = round(mean(prcsn),4),
                fscore_val = round(mean(fscore),4),
                acc_test = per_matrix_test$acc,
                recal_test = per_matrix_test$recal,
                precsn_test = per_matrix_test$precsn,
                fscore_test = per_matrix_test$fscore,
                cm_val = cm_knn_val,
                cm_test = cm_knn_test
    ))
  }
  else{
    ko = knn(x.train,x.test,y.train)
    cm.knn = table(y.test,ko)
    per_matrix = performance_eval(cm.knn)
    print(" KNN is completed...")
    
    return(list(acc=per_matrix$acc,recal=per_matrix$recal,
                precsn=per_matrix$precsn,fscore=per_matrix$fscore,
                cm = cm.knn))
  }
}

# Applying Random Forest algorithm
randomforest_mehtod = function(data.train,data.test,action){
  
  print(" Random Forest is running...")
  x.train = data.train[,1:(length(data.train)-1)]
  y.train = data.train$class_y
  x.test = data.test[,1:(length(data.test)-1)]
  y.test = data.test$class_y
  n = nrow(x.train)
  
  if(action == 'robust'){
    df = as.data.frame(cbind(x.train,y.train))
    names(df)[ncol(df)] = 'class_y'
    mtry=hyperparameter_tune('rf',df)
    K = 3
    set.seed(6405)
    folds = cut(1:n, K, labels=FALSE)
    acc = numeric(K)
    recal = numeric(K)
    prcsn = numeric(K)
    fscore = numeric(K)
    
    for(k in 1:K){
      # training sample
      i = which(folds==k)
      x.tr = x.train[-i,]
      y.tr = y.train[-i]
      # validation sample
      x.val = x.train[i,]
      y.val = y.train[i]
      # train model on training data and validation:
      rf.out = randomForest(x.tr,y.tr,mtry=mtry)
      rf.yhat = predict(rf.out, newdata=x.val, type="class")
      cm_rf_val = table(y.val,rf.yhat)
      per_matrix = performance_eval(cm_rf_val)
      acc[k] = per_matrix$acc
      recal[k] = per_matrix$recal
      prcsn[k] = per_matrix$precsn
      fscore[k] = per_matrix$fscore
    }
    
    rf.out = randomForest(x.train,y.train,mtry=mtry)
    rf.yhat = predict(rf.out, newdata=x.test, type="class")
    cm_rf_test = table(y.test,rf.yhat)
    per_matrix_test = performance_eval(cm_rf_test)
    cat('\n')
    print(" Random Forest is completed...")
    
    return(list(acc_val = round(mean(acc),4),
                recal_val = round(mean(recal),4),
                precsn_val = round(mean(prcsn),4),
                fscore_val = round(mean(fscore),4),
                acc_test = per_matrix_test$acc,
                recal_test = per_matrix_test$recal,
                precsn_test = per_matrix_test$precsn,
                fscore_test = per_matrix_test$fscore,
                cm_val = cm_rf_val,
                cm_test = cm_rf_test
    ))
  }
  else{
    rf.out = randomForest(x.train,y.train)
    rf.yhat = predict(rf.out, newdata=x.test, type="class")
    cm.rf = table(y.test,rf.yhat)
    per_matrix = performance_eval(cm.rf)
    print(" Random Forest is completed...")
    
    return(list(acc=per_matrix$acc,recal=per_matrix$recal,
                precsn=per_matrix$precsn,fscore=per_matrix$fscore,
                cm = cm.rf))
  }
}

#Applying Decision Tree algorithm
tree_method = function(data.train,data.test,action){
  
  print(" Decision Tree is running...")
  x.train = data.train[,1:(length(data.train)-1)]
  y.train = data.train$class_y
  x.test = data.test[,1:(length(data.test)-1)]
  y.test = data.test$class_y
  n = nrow(x.train)
  param = hyperparameter_tune('tree',data.train)
  K = 3
  set.seed(6405)
  folds = cut(1:n, K, labels=FALSE)
  acc = numeric(K)
  recal = numeric(K)
  prcsn = numeric(K)
  fscore = numeric(K)
  
  for(k in 1:K){
    # training sample
    i = which(folds==k)
    x.tr = x.train[-i,]
    y.tr = y.train[-i]
    # validation sample
    x.val = x.train[i,]
    y.val = y.train[i]
    # train model on training data and validation:
    tree.out = rpart(y.tr~.,data=x.tr,
                     control = rpart.control(minsplit = param$minsplit,
                                             maxdepth =param$maxdepth))
    tree.p = predict(tree.out, newdata=x.val, type="class")
    cm_tree_val = table(y.val,tree.p)
    per_matrix = performance_eval(cm_tree_val)
    acc[k] = per_matrix$acc
    recal[k] = per_matrix$recal
    prcsn[k] = per_matrix$precsn
    fscore[k] = per_matrix$fscore
  }
  
  tree.out = rpart(y.train~.,data=x.train,
                   control = rpart.control(minsplit = param$minsplit,
                                           maxdepth =param$maxdepth,
                                           cp = param$cp))
  tree.p = predict(tree.out, newdata=x.test, type="class")
  cm_tree_test = table(y.test,tree.p)
  per_matrix_test = performance_eval(cm_tree_val)
  cat('\n')
  print(" Decision Tree is completed...")
  
  return(list(acc_val = round(mean(acc),4),
              recal_val = round(mean(recal),4),
              precsn_val = round(mean(prcsn),4),
              fscore_val = round(mean(fscore),4),
              acc_test = per_matrix_test$acc,
              recal_test = per_matrix_test$recal,
              precsn_test = per_matrix_test$precsn,
              fscore_test = per_matrix_test$fscore,
              cm_val = cm_tree_val,
              cm_test = cm_tree_test
  )) 
}

# Applying SVM algorithm
SVM_mehtod = function(data.train,data.test,action){
  
  print(" SVM is running...")
  x.train = data.train[,1:(length(data.train)-1)]
  y.train = data.train$class_y
  x.test = data.test[,1:(length(data.test)-1)]
  y.test = data.test$class_y
  n = nrow(x.train)
  param = hyperparameter_tune('svm',data.train)
  cat('\n')
  K = 3
  set.seed(6405)
  folds = cut(1:n, K, labels=FALSE)
  acc = numeric(K)
  recal = numeric(K)
  prcsn = numeric(K)
  fscore = numeric(K)
  
  for(k in 1:K){
    # training sample
    i = which(folds==k)
    x.tr = x.train[-i,]
    y.tr = y.train[-i]
    # validation sample
    x.val = x.train[i,]
    y.val = y.train[i]
    # train model on training data and validation:
    svm.o = svm(y.tr~.,x.tr,kernel='linear',cost = param$cost,gamma = param$gamma)
    svm.p = predict(svm.o, newdata=x.val, type="class")
    cm_svm_val = table(y.val,svm.p)
    per_matrix = performance_eval(cm_svm_val)
    acc[k] = per_matrix$acc
    recal[k] = per_matrix$recal
    prcsn[k] = per_matrix$precsn
    fscore[k] = per_matrix$fscore
  }
  
  svm.o = svm(y.train~.,x.train,kernel='linear',cost = param$cost,gamma = param$gamma)
  svm.p = predict(svm.o, newdata=x.test, type="class")
  cm_svm_test = table(y.test,svm.p)
  per_matrix_test = performance_eval(cm_svm_test)
  print(" SVM is completed...")
  
  return(list(acc_val = round(mean(acc),4),
              recal_val = round(mean(recal),4),
              precsn_val = round(mean(prcsn),4),
              fscore_val = round(mean(fscore),4),
              acc_test = per_matrix_test$acc,
              recal_test = per_matrix_test$recal,
              precsn_test = per_matrix_test$precsn,
              fscore_test = per_matrix_test$fscore,
              cm_val = cm_svm_val,
              cm_test = cm_svm_test
  ))
}

#Create the performance matrix
performance_eval = function(cm){
  
  acc  = round(sum(diag(cm))/sum(cm),4)
  
  recal_e = cm[1,1]/sum(cm[1,])
  recal_h = cm[2,2]/sum(cm[2,])
  recal_pg = cm[3,3]/sum(cm[3,])
  recal_pm = cm[4,4]/sum(cm[4,])
  
  precision_e = cm[1,1]/sum(cm[,1])
  precision_h = cm[2,2]/sum(cm[,2])
  precision_pg = cm[3,3]/sum(cm[,3])
  precision_pm = cm[4,4]/sum(cm[,4])
  
  overall_recal = round((recal_e+recal_h+recal_pg+recal_pm)/sum(cm),4)
  
  overall_precision = round((precision_e+precision_h+precision_pg+precision_pm)/sum(cm),4)
  
  fscore = round(2*((overall_recal*overall_precision)/(overall_recal+overall_precision)),4)
  
  return(list(acc=acc,recal=overall_recal,precsn=overall_precision,fscore=fscore))
}

#Tuning the hyperparameter for different algorithms
hyperparameter_tune = function(mod,dat){
  
  if(mod=='knn'){
    print(" KNN hyperparameter tuning in progress..")
    df=dat
    lrn = makeLearner("classif.knn")
    ps = makeParamSet(makeDiscreteParam("k", values = seq(1, 25, 5)))
    ctrl = makeTuneControlGrid()
    rdesc = makeResampleDesc("CV", iters=3)
    task = makeClassifTask(data = df, target = "class_y")
    
    res = tuneParams(lrn, task = task, resampling = rdesc, 
                     par.set = ps, control =ctrl, measures = acc)
    
    cat("Optimum K value is: ",res$x$k,"\n")
    cat("Accuracy of Knn: ",res$y)
   
    return(res$x$k)
  }
  
  if(mod=='rf'){
    print(" Random Forest hyperparameter tuning in progress..")
    df=dat
    lrn = makeLearner("classif.randomForest")
    ps = makeParamSet(makeDiscreteParam("mtry", values = seq(100, 400, 50)))
    ctrl = makeTuneControlGrid()
    rdesc = makeResampleDesc("CV", iters=2)
    task = makeClassifTask(data = df, target = "class_y")
    
    res = tuneParams(lrn, task = task, resampling = rdesc,
                     par.set = ps, control =ctrl, measures = acc)
    
    cat("Optimum mtry: ",res$x$mtry,"\n")
    cat("Accuracy of RF: ",res$y)
    
    return(res$x$mtry)
  }
  if(mod=='tree'){
    print(" Decision Tree hyperparameter tuning in progress..")
    df=dat
    lrn = makeLearner("classif.rpart")
    ps = makeParamSet(makeDiscreteParam("minsplit", values = seq(10,15,2)),
                      makeDiscreteParam("maxdepth", values = seq(6,8)),
                      makeDiscreteParam("cp", values = seq(0.001, 0.002)))
    ctrl = makeTuneControlGrid()
    rdesc = makeResampleDesc("CV", iters=2)
    task = makeClassifTask(data = df, target = "class_y")
    
    res = tuneParams(lrn, task = task, resampling = rdesc,
                     par.set = ps, control =ctrl, measures = acc)
    
    cat("minsplit: ",res$x$minsplit,"\n")
    cat("maxdepth: ",res$x$maxdepth,"\n")
    cat("cp: ",res$x$cp,"\n")
    cat("Accuracy: ",res$y,"\n")
  
    return(list(minsplit=res$x$minsplit,maxdepth=res$x$maxdepth,cp=res$x$cp))
  }
  if(mod=='svm'){
    print(" SVM hyperparameter tuning in progress..")
    df=dat
    lrn = makeLearner("classif.svm")
    ps = makeParamSet(makeIntegerParam("cost", lower = 1, upper = 10),
                      makeDiscreteParam("gamma", values = c(0.25,0.5,1) ))
    ctrl = makeTuneControlGrid()
    rdesc = makeResampleDesc("CV", iters=2)
    task = makeClassifTask(data = df, target = "class_y")
    
    res = tuneParams(lrn, task = task, resampling = rdesc,
                     par.set = ps, control =ctrl, measures = acc)
    
    cat("cost: ",res$x$cost,"\n")
    cat("gamma: ",res$x$gamma,"\n")
    cat("Accuracy: ",res$y,"\n")
    
    return(list(cost=res$x$cost,gamma = res$x$gamma))
  }
}

#Extracting the important feature using Recurrsive Feature Selection algorithm
feature_selection = function(data.train,data.test){
  
  print('Feature Preprocessing is in progress...')
  x.train = data.train[,1:(length(data.train)-1)]
  y.train = data.train$class_y
  x.test = data.test[,1:(length(data.test)-1)]
  y.test = data.test$class_y
  n = nrow(x.train)
  
  set.seed(6405)
  subsets <- c(1000,2000,ncol(x.train))
  
  ctrl <- rfeControl(functions = rfFuncs,
                     method = "cv",
                     number = 3,
                     verbose = FALSE)
  
  rf.rfe <- rfe(x.train, y.train,
                sizes = subsets,
                rfeControl = ctrl)
  
  print(rf.rfe$optVariables)
  print('Feature Preprocessing is in completed...')
  
  return(rf.rfe$optVariables)
}

#Calling different algorithms and displaying the result
summary_model = function(dat_main,action){
  
  data = dat_main
  data.train = data$dat.train
  data.test = data$dat.test
  
  if(action=='basic'){
    
    nb = naive_bayes(data.train,data.test,action)
    cat('\n')
    kn = knn1(data.train,data.test,action)
    cat('\n')
    rf = randomforest_mehtod(data.train,data.test,action)
    cat('\n')
    
    print(paste0("Model is : Naive Bayes"))
    print(paste0("Confusion matrix of the model is: "))
    print(nb$cm)
    print(paste0("Accuracy of the model is: ",nb$acc))
    print(paste0("Recall of the model is: ",nb$recal))
    print(paste0("Precision of the model is: ",nb$precsn))
    print(paste0("Fscore of the model is: ",nb$fscore))
    cat('\n')
    
    print(paste0("Model is : K-Nearest Neighbours"))
    print(paste0("Confusion matrix of the model is: "))
    print(kn$cm)
    print(paste0("Accuracy of the model is: ",kn$acc))
    print(paste0("Recall of the model is: ",kn$recal))
    print(paste0("Precision of the model is: ",kn$precsn))
    print(paste0("Fscore of the model is: ",kn$fscore))
    cat('\n')
    
    print(paste0("Model is : Random Forest"))
    print(paste0("Confusion matrix of the model is: "))
    print(rf$cm)
    print(paste0("Accuracy of the model is: ",rf$acc))
    print(paste0("Recall of the model is: ",rf$recal))
    print(paste0("Precision of the model is: ",rf$precsn))
    print(paste0("Fscore of the model is: ",rf$fscore))
    cat('\n')
    
  }
  else{
    
    imp_var = feature_selection(data.train,data.test)
    data.train = data.train[,c(imp_var,'class_y')]
    data.test = data.test[,c(imp_var,'class_y')]
    
    nb = naive_bayes(data.train,data.test,action)
    cat('\n')
    kn = knn1(data.train,data.test,action)
    cat('\n')
    rf = randomforest_mehtod(data.train,data.test,action)
    cat('\n')
    tr = tree_method(data.train,data.test,action)
    cat('\n')
    svm = SVM_mehtod(data.train,data.test,action)
    cat('\n')
    
    print(paste0("Model is : Naive Bayes"))
    print(paste0("Confusion matrix of the model is: "))
    print(nb$cm_val)
    print(paste0("Accuracy of the model for validation set is: ",nb$acc_val))
    print(paste0("Recall of the model for validation set is: ",nb$recal_val))
    print(paste0("Precision of the model for validation set is: ",nb$precsn_val))
    print(paste0("Fscore of the model for validation set is: ",nb$fscore_val))
    cat('\n')
    
    print(paste0("Model is : Naive Bayes"))
    print(paste0("Confusion matrix of the model is: "))
    print(nb$cm_test)
    print(paste0("Accuracy of the model for test set is: ",nb$acc_test))
    print(paste0("Recall of the model for test set is: ",nb$recal_test))
    print(paste0("Precision of the model for test set is: ",nb$precsn_test))
    print(paste0("Fscore of the model for test set is: ",nb$fscore_test))
    cat('\n')
    
    print(paste0("Model is : K-Nearest Neighbours"))
    print(paste0("Confusion matrix of the model is: "))
    print(kn$cm_val)
    print(paste0("Accuracy of the model for validation set is: ",kn$acc_val))
    print(paste0("Recall of the model for validation set is: ",kn$recal_val))
    print(paste0("Precision of the model for validation set is: ",kn$precsn_val))
    print(paste0("Fscore of the model for validation set is: ",kn$fscore_val))
    cat('\n')
    
    print(paste0("Model is : K-Nearest Neighbours"))
    print(paste0("Confusion matrix of the model is: "))
    print(kn$cm_test)
    print(paste0("Accuracy of the model for test set is: ",kn$acc_test))
    print(paste0("Recall of the model for test set is: ",kn$recal_test))
    print(paste0("Precision of the model for test set is: ",kn$precsn_test))
    print(paste0("Fscore of the model for test set is: ",kn$fscore_test))
    cat('\n')
    
    print(paste0("Model is : Random Forest"))
    print(paste0("Confusion matrix of the model is: "))
    print(rf$cm_val)
    print(paste0("Accuracy of the model for validation set is: ",rf$acc_val))
    print(paste0("Recall of the model for validation set is: ",rf$recal_val))
    print(paste0("Precision of the model for validation set is: ",rf$precsn_val))
    print(paste0("Fscore of the model for validation set is: ",rf$fscore_val))
    cat('\n')
    
    print(paste0("Model is : Random Forest"))
    print(paste0("Confusion matrix of the model is: "))
    print(rf$cm_test)
    print(paste0("Accuracy of the model for test set is: ",rf$acc_test))
    print(paste0("Recall of the model for test set is: ",rf$recal_test))
    print(paste0("Precision of the model for test set is: ",rf$precsn_test))
    print(paste0("Fscore of the model for test set is: ",rf$fscore_test))
    cat('\n')
    
    print(paste0("Model is : Decision Tree"))
    print(paste0("Confusion matrix of the model is: "))
    print(tr$cm_val)
    print(paste0("Accuracy of the model for validation set is: ",tr$acc_val))
    print(paste0("Recall of the model for validation set is: ",tr$recal_val))
    print(paste0("Precision of the model for validation set is: ",tr$precsn_val))
    print(paste0("Fscore of the model for validation set is: ",tr$fscore_val))
    cat('\n')
    
    print(paste0("Model is : Decision Tree"))
    print(paste0("Confusion matrix of the model is: "))
    print(tr$cm_test)
    print(paste0("Accuracy of the model for test set is: ",tr$acc_test))
    print(paste0("Recall of the model for test set is: ",tr$recal_test))
    print(paste0("Precision of the model for test set is: ",tr$precsn_test))
    print(paste0("Fscore of the model for test set is: ",tr$fscore_test))
    cat('\n')
    
    print(paste0("Model is : SVM"))
    print(paste0("Confusion matrix of the model is: "))
    print(svm$cm_val)
    print(paste0("Accuracy of the model for validation set is: ",svm$acc_val))
    print(paste0("Recall of the model for validation set is: ",svm$recal_val))
    print(paste0("Precision of the model for validation set is: ",svm$precsn_val))
    print(paste0("Fscore of the model for validation set is: ",svm$fscore_val))
    cat('\n')
    
    print(paste0("Model is : SVM"))
    print(paste0("Confusion matrix of the model is: "))
    print(svm$cm_test)
    print(paste0("Accuracy of the model for test set is: ",svm$acc_test))
    print(paste0("Recall of the model for test set is: ",svm$recal_test))
    print(paste0("Precision of the model for test set is: ",svm$precsn_test))
    print(paste0("Fscore of the model for test set is: ",svm$fscore_test))
  }
}

#Based on the action it will do the evaluation by calling respective functions 
evaluation = function(action){
  
  #Reading the files by calling the read_data function
  data=read_data()
  
  #Writing the words, their frequency and class in csv file
  word_cnt_fname = eval_count(data$dat.train,action,'train')
  word_cnt_fname = eval_count(data$dat.test,action,'test')
  
  #Read te data by calling the read_split_data and create the bag of words
  dat_main = read_data_csv()
  
  #Split the data into test and validation set and call differernt ML techniques
  summary_model(dat_main,action)
}

# Take the input from user
user_input = function(){
  
  inp_eval = 0
  par(mfrow=c(2,4))
  
  print(paste0("Enter the type of Evaluation "))
  print("Enter 1 for 200 most popular words")      
  print("Enter 2 for 200 most popular words by range")
  print("Enter 3 for Basic Evaluation")
  print("Enter 4 for Robust Evaluation ")
  inp_eval = as.integer(readline(prompt="Enter the input: "))
  
  action = 'robust'
  
  if(inp_eval == 1){
    mostpopular200()
  }
  if(inp_eval == 2){
    words_rnge()
  }
  if(inp_eval == 3){
    action = 'basic'
    evaluation(action)
  }
  if(inp_eval == 4){
    action = 'robust'
    evaluation(action)
  }
}

#######################################################################################
#*************************************EVALUATION***************************************
#######################################################################################
#Run this step after calling all of the above functions
action = user_input()

#######################################################################################
#*************************************EVALUATION***************************************
#######################################################################################
