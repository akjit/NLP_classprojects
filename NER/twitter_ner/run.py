#!/usr/bin/env python

'''Spanish NER tagger using HMM's'''
'''Abhijit Kulkarni, CS 190I '''


import os
import io
import sys
import math
import numpy as np
import csv
import codecs
import collections
import random
import time
import pickle

def readData(filename):
	multi_dim_data = []
	curr_sentence = []
	f = codecs.open(filename, "r")
	count = 0
	for line in f.readlines():
		if line == " \n":
			multi_dim_data.append(curr_sentence)
			curr_sentence = []
		else:
			curr_sentence.append(tuple(line.strip().split(" ")))
			count = count + 1
	multi_dim_data.append(curr_sentence)
	np_data = np.array(multi_dim_data)
	return np_data

def readTestData(filename, words):
	multi_dim_data = []
	curr_sentence = []
	unknowns = []

	f = codecs.open(filename,"r")

	for line in f.readlines():
		if line == "\n":
			multi_dim_data.append(curr_sentence)
			curr_sentence = []
		else:
			curr_sentence.append(line.strip())
			if line.strip() not in words:
				unknowns.append(line.strip())

	multi_dim_data.append(curr_sentence)
	np_data = np.array(multi_dim_data)
	return np_data, unknowns

def findStart(data, states):
	start_p = np.array([0.0]*len(states))
	for i in range(len(data)):
		index = states.index(data[i][0][1])
		start_p[index] += 1.0
	start_p = start_p / float(len(data))
	return start_p
	
def findTransition(data, states):
	oneD_data = np.concatenate(data)
	num_states = len(states)
	trans_p = np.array([[0.0]*num_states for i in range(num_states)])
	num_first_tag = np.array([0.0]*num_states)	#number of times this tag was the #1 in 1->2

	for i in range(len(oneD_data)-1):
		s1 = states.index(oneD_data[i][1])
		s2 = states.index(oneD_data[i+1][1])
		num_first_tag[s1] += 1.0
		trans_p[s1][s2] += 1.0

	for i in range(num_states):
		if num_first_tag[i] != 0:
			trans_p[i] = trans_p[i] / num_first_tag[i]

	return trans_p

def findEmission(data, states):
	oneD_data = np.concatenate(data)
	num_states = len(states)
	words = dict()
	words['UNK##'] = [0.0]*num_states
	state_occurrences = np.array([0.0]*num_states)

	unknowns = []

	for i in range(len(oneD_data)):
		curr_word = oneD_data[i][0]
		curr_state = states.index(oneD_data[i][1])
		if curr_word in unknowns:
			curr_word = "UNK##"
		if curr_word not in words:
			if random.random() <= 0.01:
				unknowns.append(curr_word)
				curr_word = "UNK##"
			else:
				words[curr_word] = [0.0]*num_states

		if words.get(curr_word)[curr_state] == 0.0:
			words[curr_word][curr_state] = 1.0
			state_occurrences[curr_state] += 1.0
		else:
			words[curr_word][curr_state] = words[curr_word][curr_state] + 1.0
			state_occurrences[curr_state] += 1.0
	# c =1
	for i in range(num_states):
		for word in words:
			if state_occurrences[i] != 0:
				words[word][i] = words[word][i] / state_occurrences[i]
	# print words["UNK##"]
	# print len(unknowns)
	# print c
	return words

def viterbi(test_data, states, trans_p, start_p, emis_p, unknowns,count):
	obs = np.copy(test_data)
	vit = np.array([[0.0]*len(obs) for row in range(len(states))])
	bp = {}
	for i in range(len(states)):
	 	# vit[i][0] = start_p[i] * emis_p[obs[0][0]][i]
	 	if obs[0] in unknowns:
	 		vit[i][0] = start_p[i] * emis_p["UNK##"][i]
	 	else:
	 		vit[i][0] = start_p[i] * emis_p[obs[0]][i]
	 	bp[i] = [0]
	for t in range(1,len(obs)):
		temp_bp = {}
		count = count + 1
		# if random.random() <= 0.01:
		# 	print "Word Number: " + str(count)
		for i in range(len(states)):
			# (p,s) = max((vit[s_iter][t-1] * trans_p[s_iter][i] * emis_p[obs[t][0]][i], s_iter) for s_iter in range(len(states)))
			#(p,s) = max((vit[s_iter][t-1] * trans_p[s_iter][i] * emis_p[obs[t]][i], s_iter) for s_iter in range(len(states)))
			maxp = -1
			maxs = -1
			for s_iter in range(len(states)):
				if obs[t] in unknowns:
					p = vit[s_iter][t-1] * trans_p[s_iter][i] * emis_p["UNK##"][i]
				else:
					p = vit[s_iter][t-1] * trans_p[s_iter][i] * emis_p[obs[t]][i]					
				if p > maxp:
					maxp = p
					maxs = s_iter
			vit[i][t] = maxp
			temp_bp[i] = bp[maxs]+[i]
		bp = temp_bp

	maxp = -1
	maxs = -1
	for s_iter in range(len(states)):
		p = vit[i][len(obs) -1]
		if p > maxp:
			maxp = p
			maxs = s_iter
	return (bp[maxs],count)

# def forward(validation_data, states, trans_p, start_p, emis_p, unknowns):
# 	obs = np.copy(validation_data)
# 	fwd = np.array([[0.0]*len(obs) for row in range(len(states))])

# 	for i in range(len(states)):
# 		if obs[0,0] in unknowns or obs[0,0] not in emis_p:
# 			fwd[i][0] = start_p[i] * emis_p["UNK##"][i]
# 		else:
# 			fwd[i][0] = start_p[i] * emis_p[obs[0,0]][i]

# 	for t in range(len(obs)):
# 		for i in range(len(states)):
# 			if obs[t,0] in unknowns or obs[t,0] not in emis_p:
# 				fwd[i][t] = sum((fwd[s_iter][t-1] * trans_p[s_iter][i] * emis_p["UNK##"][i]) for s_iter in range(len(states)))
# 			else:
# 				fwd[i][t] = sum((fwd[s_iter][t-1] * trans_p[s_iter][i] * emis_p[obs[t,0]][i]) for s_iter in range(len(states)))


# 	fwd_p = sum((fwd[s][len(obs)-1]) for s in range(len(states)))
# 	return (fwd, fwd_p)

#def backward():

#def forward_backward():


#def tuneParameters(validation_data, states, trans_p, start_p, emis_p):

#def run(train_file, validation_file, test_file, pred_file):
def run(test_file, pred_file):
	# train = readData(train_file)
	# train_data = readData(train_file)
	# validation_data = readData(validation_file)

	states = ["O", "B-musicartist", "I-musicartist", "B-sportsteam", "I-sportsteam", "B-product", "I-product", "B-geo-loc", "I-geo-loc", "B-movie", "I-movie", "B-tvshow", "I-tvshow", "B-company", "I-company", "B-person", "I-person", "B-facility", "I-facility", "B-other", "I-other"]
	# start_p = findStart(train_data,states)
	# print zip(states,start_p)
	# trans_p = findTransition(train_data, states)
	# emis_p = findEmission(train_data, states)
	start_p = np.load("start.npy")
	trans_p = np.load("trans.npy")
	with open ("emis.txt", 'rb') as handle:
		emis_p = pickle.loads(handle.read())
	traint = time.time() - start
	#print traint 
	test_data, unknowns = readTestData(test_file, emis_p)
	#print len(test_data)
	final_predictions = []

	# np.save("start",start_p)
	# np.save("trans", trans_p)
	# with open("emis.txt", 'wb') as handle:
	# 	pickle.dump(emis_p, handle)

	# true_pred = []
	# for sentence in range(len(validation_data)):
	# 	tags = viterbi(validation_data[sentence], states, trans_p, start_p, emis_p)
	# 	for i in range(len(tags)):
	# 		final_predictions.append(states[tags[i]])
	ct = 0
	for sentence in range(len(test_data)):
		tags,ct = viterbi(test_data[sentence], states, trans_p, start_p, emis_p, unknowns,ct)
		for i in range(len(tags)):
			final_predictions.append(states[tags[i]])

	writer = open(pred_file, "w")
	count = 0
	for sentence in range(len(test_data)):
		for word in range(len(test_data[sentence])):
			writer.write(test_data[sentence][word] + " " + final_predictions[count] + "\n")
			count = count + 1
		if sentence != len(test_data) -1:
			writer.write("\n")
	writer.close()
	endt = time.time() - start
	# print "Training: " + str(traint)
	# print "Testing: " + str(endt)
		
	# print final_predictions
	# for sentence in range(len(validation_data)):
	# 	for word in range(len(validation_data[sentence])):
	# 		true_pred.append(validation_data[sentence][word][1])

	# correct = 0.0
	# for i in range(len(true_pred)):
	# 	if true_pred[i] == final_predictions[i]:
	# 		correct = correct + 1.0
	# accuracy = correct / float(len(true_pred))
	# #print true_pred
	# print "\nACCURACY: " + str(accuracy)
	#print emis_p



if __name__ == "__main__":
	# if len(sys.argv) != 5:
	# 	print("Received wrong number of arguments. \nUsage: \'python run.py train validation test pred ")
	# 	sys.exit()
	start = time.time()
	#run(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4])
	run(sys.argv[1], sys.argv[2])