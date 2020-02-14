#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:06:52 2019

@author: vanshsmacpro
"""

import os
import random
import copy 
import csv
import numpy as np
import timeit
import matplotlib.pyplot as plt
from collections import Counter
import sys

def read_data(fname):
    
    file = open("20newsgroups/" + fname , 'r');
    for line in file:
        x = line.split(" ")
        
   

    return x



def read(directory):
    y = []
    order_file_opened = []
    for filename in os.listdir(directory):
       
        if filename != "index.csv" :
            
            x = read_data(filename)
            order_file_opened.append(filename)
            #print(filename)
            y.append(x)
    return y, order_file_opened
            
        

# remove blank spaces of last element in list 
def refine(y):
    count = 0 
    bag_of_words  = []     
    for i in range(len(y)):
        for j in range(len(y[i])):
            if j == len(y[i]) - 1 : 
                del(y[i][j])
            else : 
                count = count + 1
                bag_of_words.append(y[i][j])
    unique_words = []           
    for i in bag_of_words : 
        if i not in unique_words : 
            unique_words.append(i)
    return bag_of_words ,unique_words, y
                
def create_csv(lists):
    with open('topicwords.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile)
     for i in range(K):
         wr.writerow([i ,lists[i]])
def create_csv1(lists ):
    with open('final_c_d.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile)
     for i in range(K):
         wr.writerow(lists[i])
         
def split_data(train_data , train_label):
    matrix_data = []
    matrix_label = []
    n = int(0.1*len(train_label))
    x = 0.1*len(train_label)
    count = 0 
    flag = 1
    while n <= len(train_label) : 
        traini = []
        testi = []
        for i in range(n):
            
            traini.append(train_data[count])
            testi.append(train_label[count])
            count = count + 1
            
            if count == (n-1):
                flag = flag + 1
                n = int( flag*x  )
                count = 0  
                
        matrix_data.append(traini)
        matrix_label.append(testi)
    return matrix_data , matrix_label

def optimization_technique(d,r,phi,w):
    phi_t = phi.T
    alpha = 0.01
    I = np.identity(len(phi[0]), dtype = float)
    H_without_i = phi_t.dot(r)
    H_without_i = -H_without_i.dot(phi) - I.dot(alpha)
    H_inverse = np.linalg.inv(H_without_i)
    g = phi_t.dot(d) - w.dot(alpha)
    
    new_w = w - H_inverse.dot(g)
    return new_w
    
# logistic method to calculate W
def calculate_train_logistic(matrix_data, matrix_label , number_features):
    w = number_features*[0]
    w = np.array(w)
    w_all = []
    position = []
    time_for_all = []
    start = 0 
    for j in range(len(matrix_data)):
        for i in range(500):
        
            phi = matrix_data[j]
            phi = np.array(phi)
        
        
            y_without_logi = phi.dot(w)
            y = 1/(1 + np.exp(-y_without_logi))
            t = matrix_label[j]
            d = t - y
            r = y.dot(1-y)
            new_w = optimization_technique(d,r,phi,w) 
            temp = w
            w = new_w
            stopping_condition = np.sqrt(np.sum(np.square(w - temp)))/np.sqrt(np.sum(np.square(temp)))
            
            if stopping_condition <= 0.001 or i == 499 : 
                position.append(i)
                end = timeit.default_timer()
                time = end - start
                time_for_all.append(time)
                #print(i)
                break
            
                
           
           
        w_all.append(w)
    return w_all , position , time_for_all

def calculate_test_logistic(test_data, test_label,learning_parameter_w):   
    error_all = []
    for j in range(len(learning_parameter_w)):
        error = 0
    
        test_without_sigmoid = test_data.dot(learning_parameter_w[j])
        predicted = 1/(1+np.exp(-test_without_sigmoid))
        predicted_final = []
        for i in range(len(predicted)):
            if predicted[i] >= 0.5:
                predicted_final.append(1)
            else:
                predicted_final.append(0)
    
        for i in range(len(predicted)):
            if predicted_final[i] != test_label[i]:
                error = error + 1
        error_all.append(1 - (error/len(test_label)))
        
    return error_all


def calculate_mean_sd(mean_sd):
    count = 0
    len(mean_sd)
    mean_same_train = []
    mean_sd_same = []
    while(count < len(mean_sd[0])):
        for i in range(len(mean_sd)) : 
            x = mean_sd[i]
            for j in range(len(mean_sd[0])):
                if count == j : 
                    mean_same_train.append(x[j])
                    
        count = count + 1
        mean_sd_same.append(mean_same_train)
        mean_same_train = []
    #print(mean_sd_same)                
    
    
    mean_all = []
    sd_all = []
    
    for i in mean_sd_same:
        mean = np.mean(i)
        mean_all.append(mean)
        sd = np.std(i)
        sd_all.append(sd)
    
    return mean_all , sd_all

# fucntion to plot graph 

def plot_graph(mean_all , sd_all , method1 , mean_all_1 , sd_all_1 , method2):
    x = [0.1, 0.2, 0.3 , 0.4, 0.5 , 0.6, 0.7, 0.8, 0.9 , 1.0]
    z = "learning curve for " + method1
    z2 = "learning curve for" + method2
    plt.errorbar(x, mean_all,sd_all, label = z)
    plt.errorbar(x , mean_all_1 , sd_all_1 , label = z2)
    plt.xlabel("increasing data size")
    plt.ylabel("accuracy")
    plt.legend(loc = "lower right")
    plt.show()
    
def logistic(final_c_d , final_label1 , alpha1, flag):   
    data = []
    if flag == 1 : 
        
        for i in range(D):
            pic = []
            for j in range(K):
                pic.append(0)
            data.append(pic)
        for i in range(D):
            for j in range(K):
                num1 = final_c_d[i][j] + alpha1
                den1 = K*alpha1 + sum(final_c_d[i])
                data[i][j] = num1/den1
        
    else : 
        data = final_c_d
    data = np.array(data)
    final_label1 = np.array(final_label1)
    number_features = len(data[0])
    mean_sd = []
   
    
    for i in range(30):
        index = np.random.permutation(len(final_label1))
        data , final_label1 =  data[index] , final_label1[index]
    
    #print(data, label)
    
        x = int(len(data)*2/3)
        train_label = final_label1[:x]
        test_label = final_label1[x:]
        train_data = data[:x]
        test_data = data[x:]
        matrix_data , matrix_label = split_data(train_data ,train_label )
        learning_parameter_w, position , time_for_all  = calculate_train_logistic(matrix_data, matrix_label, number_features)
                
        
    
        error_all = calculate_test_logistic(test_data, test_label,learning_parameter_w )
        mean_sd.append(error_all)
    mean_all , sd_all = calculate_mean_sd(mean_sd)
    return mean_all , sd_all
    
    
def writing_in_csv(unique_words , final_c_t ):
    
    lists = []
    for i in range(len(final_c_t)):
        copy1 = copy.deepcopy(final_c_t[i])
        copy1.sort(reverse = True)
        x = copy.deepcopy(copy1)


       
        max1 = x[0]
        indexx1 = final_c_t[i].index(max1)
        tags1 = unique_words[indexx1]
        
        max2 = x[1]
        indexx2 = final_c_t[i].index(max2)
        tags2 = unique_words[indexx2]
        
        max3 = x[2]
        indexx3 = final_c_t[i].index(max3)
        tags3 = unique_words[indexx3]
        
        max4 = x[3]
        indexx4 = final_c_t[i].index(max4)
        tags4 = unique_words[indexx4]
        
        max5 = x[4]
        indexx5 = final_c_t[i].index(max5)
        tags5 = unique_words[indexx5]
        listt = [tags1, tags2,tags3,tags4,tags5]
        lists.append(listt) 
        listt = []
    create_csv(lists)
    
def gibbs_sampling(final_c_d,final_c_t,bag_of_words,unique_words ,pi):
    probabilities = [] 
    N_iteration = 500
    beta = 0.01
    K = 20
    alpha = 5/K
    for j in range(K):
        probabilities.append(0)
   
    for i in range(N_iteration):
        print(i)
        
        for j in range(len(bag_of_words)):
             
            
            ind = pi[j]
            word_at_ind = bag_of_words[ind]
            word = unique_words.index(word_at_ind)
            #print(word)
            document = doc[pi[j]]
            #print(document)
            topic = topic_all[pi[j]]
            #print(topic)
            final_c_d[document][topic] = final_c_d[document][topic] - 1
            #print(final_c_d[document][topic])
            
            final_c_t[topic][word] = final_c_t[topic][word] - 1
            for k in range(K):
                nume1 = final_c_t[k][word] + beta
                summ1 = sum(final_c_t[k])
                   
                denom1 = V*beta + summ1
                
                nume2 = final_c_d[document][k] + alpha
                
                
                
                summ2 = sum(final_c_d[document])
            
                denom2 = K*alpha + summ2 
                
                probabilities[k] = (nume1*nume2)/(denom1*denom2)
                
                    
                #print( (nume1*nume2)/(denom1*denom2))
                #print(probabilities[k])
            # normalize p
            x = sum(probabilities)
            probabilities = list(np.array(probabilities)/x)
            accumulated = 0  
            # topic <- sample from p
            r = random.uniform(0,1)
            
            for k in range(len(probabilities)):
                
                accumulated = sum(probabilities[:k+1])
                if r <= accumulated : 
                    index = k
                    break
                
            topic = index
                    
            topic_all[pi[j]] = topic 
            final_c_d[document][topic] = final_c_d[document][topic] + 1
            final_c_t[topic][word] = final_c_t[topic][word] + 1
    return final_c_d , final_c_t ,  topic_all

def c_t(K,V, bag_of_words, unique_words, tags,topic_all):    
    final_c_t = []
    for i in range(K):
        c_d = []
        for j in range(V):
            c_d.append(0)
        final_c_t.append(c_d)
    
    
    for k in range(len(bag_of_words)):
        for j in range(V):
            if bag_of_words[k] == unique_words[j]:
                for i in range(len(tags)):
                    if topic_all[k] == tags[i] : 
                        final_c_t[i][j] = final_c_t[i][j] + 1
                        break
    return final_c_t

def c_d(D,K,topics,tags) : 
    final_c_d = []    
    for i in range(D):
        c_d = []
        for j in range(K):
            c_d.append(0)
        final_c_d.append(c_d)
    
    for i in range(len(topics)):
        
        
        
        for j in range(len(topics[i])):
            for k in range(len(tags)):
                if topics[i][j] == tags[k] : 
                    final_c_d[i][k] = final_c_d[i][k] + 1
                    break
    return final_c_d
        
def second_method(D , V, y , unique_words):
    second_part = []
    for i in range(D):
        si = []
        for j in range(V):
            si.append(0)
        second_part.append(si)
    
    # check for all documents
        
    for i in range(len(y)):
        for j in range(len(y[i])):
            for k in range(len(unique_words)) : 
                if y[i][j] == unique_words[k] : 
                    second_part[i][k] = second_part[i][k] + 1
                    break
    return second_part
                       


###### main 
            
s = sys.argv[1]
directory = s
y , order_file_opened = read(directory)
bag_of_words ,unique_words, y = refine(y)
#print(unique_words)
V = len(unique_words)
D = 200
K = 20
# shuffling and genrating sequence of pi(n)
pi = []
for i in range(len(bag_of_words)):
    pi.append(i)
random.shuffle(pi)

# initializing d(n)
doc = []
for i in range(len(y)):
    for j in range(len(y[i])):
        doc.append(i)
alpha = 5/K        
# intializing topic 
topic_all = [] 
topics = []
for i in range(len(y)):
    top = []
    for j in range(len(y[i])):
        r = random.randint(0,19)
        topic_all.append(r)
        top.append(r)
    topics.append(top)
                    
       
    
countsss = Counter(bag_of_words) 
len(countsss)  


tags = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
final_c_d = c_d(D,K,topics,tags)
     
# initialize ct with K X V matrix 

final_c_t = c_t(K,V, bag_of_words, unique_words, tags, topic_all)
final_c_d , final_c_t ,  topic_all = gibbs_sampling(final_c_d,final_c_t,bag_of_words,unique_words ,pi)

writing_in_csv(unique_words , final_c_t)



label = np.genfromtxt(s + "/index.csv" , delimiter = ",")

final_label1 = []
for i in range(len(order_file_opened)):
    
    final_label1.append(label[int(order_file_opened[i])- 1][1])
    
mean_all , sd_all = logistic(final_c_d , final_label1, alpha, 1)

data_2 = second_method(D , V, y , unique_words)

mean_all_1 , sd_all_1 = logistic(data_2 , final_label1 , alpha, 0)

method1 = "LDA"
method2= "bag_of_words"
plot_graph(mean_all , sd_all , method1, mean_all_1 , sd_all_1 , method2)
    

        



    
