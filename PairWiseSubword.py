import numpy as np
import tensorflow as tf
import math
import logging
import time
import os
import sys
import cPickle
import argparse
from datetime import datetime
from collections import OrderedDict

from pw_util import readURLdata,URL_maxF1_eval

logger = logging.getLogger('pairwise-subword')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)


def param_init(Config):

	granularity = Config.granularity
	lm_mode = Config.lm_mode
	embedding_dim = Config.embedding_dim
	hidden_dim = Config.hidden_dim
	tokens = Config.tokens
	params = OrderedDict()

	if granularity == 'char':
		params['df'] = tf.Variable(tf.random_uniform(embedding_dim, embedding_dim))
		params['db'] = tf.Variable(tf.random_uniform(embedding_dim, embedding_dim))
		params['bias'] = tf.Variable(tf.random_uniform(embedding_dim))
		params['Wx'] = tf.Variable(tf.random_uniform(embedding_dim, embedding_dim))
		params['W1'] = tf.Variable(tf.random_uniform(embedding_dim, embedding_dim))
		params['W2'] = tf.Variable(tf.random_uniform(embedding_dim, embedding_dim))
		params['W3'] = tf.Variable(tf.random_uniform(embedding_dim, embedding_dim))
		params['vg'] = tf.Variable(tf.random_uniform(embedding_dim, 1))
		params['bg'] = tf.Variable(tf.random_uniform(1, 1))

		if lm_mode:
			params['lm_Wm_forward'] = tf.Variable(tf.random_uniform(hidden_dim,hidden_dim))
			params['lm_Wm_backward'] = tf.Variable(tf.random_uniform(hidden_dim,hidden_dim))
			params['lm_Wq_forward'] = tf.Variable(tf.random_uniform(hidden_dim,len(tokens)))
			params['lm_Wq_backward'] = tf.Variable(tf.random_uniform(hidden_dim,len(tokens)))

	return params


def load_text_vec(fname):
    vectors = {}
    with open(fname, "r") as f:
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    return vectors


def deep_CNN_layers(Config,kernel_num,is_char):

        CNN_layer = tf.keras.Sequential()
	if is_char:
		feature_maps = Config.char_maps[kernel_num]
		charCNN_len = Config.charCNN_max_word_length
		kernel_sz = (Config.char_kernels[kernel_num],Config.charCNN_embedding_size)
		conv2d = tf.keras.layers.Conv2D(filters=feature_maps, kernel_size=kernel_sz, activation='tanh', strides=(1,1),
		padding="VALID")
		maxpool2d = tf.keras.layers.MaxPool2D(pool_size=(charCNN_len - kernel_sz[0] + 1,1), strides=1)
                CNN_layer.add(conv2d)
                CNN_layer.add(maxpool2d)
	else:
		feature_maps = Config.word_maps[kernel_num]
		kernel_sz = (Config.kernels[kernel_num],Config.kernels[kernel_num])
		conv2d = tf.keras.layers.Conv2D(filters=feature_maps, kernel_size=kernel_sz, activation='relu', strides=(1,1), 
		padding="SAME", kernel_initializer=tf.keras.initializers.RandomNormal(stddev = math.sqrt(2/(kernel_sz[0]*kernel_sz[1]*feature_maps))))
		if kernel_num == 5:
			p_size = 3
			stride = 2
		else:
			p_size = 2
			stride = 1
		maxpool2d = tf.keras.layers.MaxPool2D(pool_size=(p_size,p_size), strides=stride)
                CNN_layer.add(conv2d)
                CNN_layer.add(maxpool2d)
	
	return CNN_layer


class Config:
	num_class = 2
	hidden_dim = 250
	num_layers = 1
	char_maps = [50, 100, 150, 200, 200, 200, 200]
	word_maps = [128, 164, 192, 192, 128, 128]
	kernels = [1, 2, 3, 4, 5, 6, 7]
	charcnn_max_word_length = 20
	charcnn_embedding_size = 15
	combine_mode = None

	# valid_batch_size = 10


	def __init__(self,granularity,deep_CNN,word_mode,dict_char_ngram,word_freq,oov,tokens,lm_mode,EMBEDDING_DIM):
		self.granularity = granularity
                self.deep_CNN = deep_CNN
		self.tokens = tokens
		self.dict_char_ngram = dict_char_ngram
		self.word_freq = word_freq
		self.oov = oov
		self.word_mode = word_mode
		self.lm_mode = lm_mode
		self.embedding_dim = EMBEDDING_DIM

		word2id = {}
		index = 0
		for word in tokens:
			word2id[word] = index
			index += 1
		self.word2id = word2id


class DeepPairWiseWord():

	def __init__(self,Config,true_dict,fake_dict):
		self.Config = Config
		self.true_dict = true_dict
		self.fake_dict = fake_dict
                self.layer = []

		if Config.granularity == 'char':
			self.c2w_embedding = tf.keras.layers.Embedding(len(dict_char_ngram),50)
			self.char_cnn_embedding = tf.keras.layers.Embedding(len(dict_char_ngram),Config.charcnn_embedding_size)
			self.bi_lstm_c2w = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences = True),merge_mode='concat')

			for i in range(7):
				self.charCNN_filter[i] = deep_CNN_layers(self.Config,i,True)


			self.transform_gate = tf.keras.layers.Dense(1100, activation='sigmoid')
			self.char_cnn_mlp = tf.keras.layers.Dense(1100, activation='tanh')
			self.down_sampling_200 = tf.keras.layers.Dense(200)
			self.down_sampling_300 = tf.keras.layers.Dense(300)

		# elif granularity == 'word':

		self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(Config.hidden_dim,return_sequences = True),merge_mode='concat')

		if not Config.deep_CNN:
			self.mlp_layer = tf.keras.Sequential()
			self.mlp_layer.add(tf.keras.layers.Dense(16))
			self.mlp_layer.add(tf.keras.layers.Dense(Config.num_class,activation='softmax'))
		else:
			for i in range(6):
				self.layer.append(deep_CNN_layers(self.Config,i,False))
			self.fc1 = tf.keras.layers.Dense(128,activation='relu',kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1),bias_initializer='zero')
			self.fc2 = tf.keras.layers.Dense(Config.num_class,activation='softmax',kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1),bias_initializer='zero')
		self.params = param_init(Config)
		self.inputPlaceholder_A = tf.placeholder(tf.float32,[/])
		self.build()


	def unpack(self,bi_hidden, half_dim):
		# print(bi_hidden[0])
		for i in range(bi_hidden.get_shape().as_list()[0]):
			vec = bi_hidden[i][:]
			if i == 0:
				h_fw = tf.reshape(vec[:half_dim],(1,-1))
				h_bw = tf.reshape(vec[half_dim:],(1,-1))
			else:
				h_fw_new = tf.reshape(vec[:half_dim],(1, -1))
				h_bw_new = tf.reshape(vec[half_dim:],(1, -1))
				h_fw = tf.concat((h_fw,h_fw_new),0)
				h_bw = tf.concat((h_bw,h_bw_new),0)
		# print(h_fw.size())
		# print(h_fw[0])
		# print(h_bw[0])
		# sys.exit()
		return (h_fw, h_bw)


	def pairwise_word_interaction(self,out0,out1, target_A, target_B):

			size00 = out0.get_shape().as_list()[0]
			size02 = out0.get_shape().as_list()[2]
			size10 = out1.get_shape().as_list()[0]
			size12 = out1.get_shape().as_list()[2]

			h_fw_0, h_bw_0 = self.unpack(tf.reshape(out0,(size00,size02)),half_dim=self.Config.hidden_dim)
			h_fw_1, h_bw_1 = self.unpack(tf.reshape(out1,(size10,size12)),half_dim=self.Config.hidden_dim)
			# print(h_fw_0)
			# print(h_bw_0)
			# print(h_fw_1)
			# print(h_bw_1)
			# sys.exit()
			h_bi_0 = tf.reshape(out0,(size00,size02))
			h_bi_1 = tf.reshape(out1,(size10,size12))
			h_sum_0 = h_fw_0 + h_bw_0
			h_sum_1 = h_fw_1 + h_bw_1
			len0 = h_fw_0.get_shape().as_list()[0]
			len1 = h_fw_1.get_shape().as_list()[0]
			i = 0
			j = 0
			# simCube1 = tf.matmul(h_fw_0[i].view(1, -1), h_fw_1[j].view(-1, 1))
			# simCube2 = tf.matmul(h_bw_0[i].view(1, -1), h_bw_1[j].view(-1, 1))
			# simCube3 = tf.matmul(h_bi_0[i].view(1, -1), h_bi_1[j].view(-1, 1))
			# simCube4 = tf.matmul(h_sum_0[i].view(1, -1), h_sum_1[j].view(-1, 1))
			# simCube5 = F.pairwise_distance(h_fw_0[i].view(1, -1), h_fw_1[j].view(1, -1))
			simCube5_0 = tf.reshape(h_fw_0[i],(1, -1))
			simCube5_1 = tf.reshape(h_fw_1[j],(1, -1))
			# simCube6 = F.pairwise_distance(h_bw_0[i].view(1, -1), h_bw_1[j].view(1, -1))
			simCube6_0 = tf.reshape(h_bw_0[i],(1, -1))
			simCube6_1 = tf.reshape(h_bw_1[j],(1, -1))
			# simCube7 = F.pairwise_distance(h_bi_0[i].view(1, -1), h_bi_1[j].view(1, -1))
			simCube7_0 = tf.reshape(h_bi_0[i],(1, -1))
			simCube7_1 = tf.reshape(h_bi_1[j],(1, -1))
			# simCube8 = F.pairwise_distance(h_sum_0[i].view(1, -1), h_sum_1[j].view(1, -1))
			simCube8_0 = tf.reshape(h_sum_0[i],(1,-1))
			simCube8_1 = tf.reshape(h_sum_1[j],(1,-1))
			# simCube9 = F.cosine_similarity(h_fw_0[i].view(1, -1), h_fw_1[j].view(1, -1))
			# simCube10 = F.cosine_similarity(h_bw_0[i].view(1, -1), h_bw_1[j].view(1, -1))
			# simCube11 = F.cosine_similarity(h_bi_0[i].view(1, -1), h_bi_1[j].view(1, -1))
			# simCube12 = F.cosine_similarity(h_sum_0[i].view(1, -1), h_sum_1[j].view(1, -1))
			for i in range(len0):
				for j in range(len1):
					if not(i == 0 and j == 0):
						simCube5_0 = tf.concat((simCube5_0, tf.reshape(h_fw_0[i],(1, -1))),0)
						simCube5_1 = tf.concat((simCube5_1, tf.reshape(h_fw_1[j],(1, -1))),0)
						simCube6_0 = tf.concat((simCube6_0, tf.reshape(h_bw_0[i],(1, -1))),0)
						simCube6_1 = tf.concat((simCube6_1, tf.reshape(h_bw_1[j],(1, -1))),0)
						simCube7_0 = tf.concat((simCube7_0, tf.reshape(h_bi_0[i],(1, -1))),0)
						simCube7_1 = tf.concat((simCube7_1, tf.reshape(h_bi_1[j],(1, -1))),0)
						simCube8_0 = tf.concat((simCube8_0, tf.reshape(h_sum_0[i],(1, -1))),0)
						simCube8_1 = tf.concat((simCube8_1, tf.reshape(h_sum_1[j],(1, -1))),0)

			simCube1 = tf.matmul(h_fw_0, tf.transpose(h_fw_1, 0, 1))
			simCube2 = tf.matmul(h_bw_0, tf.transpose(h_bw_1, 0, 1))
			simCube3 = tf.matmul(h_bi_0, tf.transpose(h_bi_1, 0, 1))
			simCube4 = tf.matmul(h_sum_0, tf.transpose(h_sum_1, 0, 1))

			simCube5 = tf.negative(tf.norm(simCube5_0 - simCube5_1),1)
			simCube5 = tf.reshape(simCube5,(len0, len1))
			simCube6 = tf.negative(tf.norm(simCube6_0 - simCube6_1),1)
			simCube6 = tf.reshape(simCube6,(len0, len1))
			simCube7 = tf.negative(tf.norm(simCube7_0 - simCube7_1),1)
			simCube7 = tf.reshape(simCube7,(len0, len1))
			simCube8 = tf.negative(tf.norm(simCube8_0 - simCube8_1),1)
			simCube8 = tf.reshape(simCube8,(len0, len1))

			simCube9 = tf.losses.cosine_distance(simCube5_0,simCube5_1,axis=1,reduction=None)
			simCube9 = tf.reshape(simCube9,(len0,len1))
			simCube10 = tf.losses.cosine_distance(simCube5_0,simCube5_1,axis=1,reduction=None)
			simCube10 = tf.reshape(simCube10,(len0,len1))
			simCube11 = tf.losses.cosine_distance(simCube5_0,simCube5_1,axis=1,reduction=None)
			simCube11 = tf.reshape(simCube11,(len0,len1))
			simCube12 = tf.losses.cosine_distance(simCube5_0,simCube5_1,axis=1,reduction=None)
			simCube12 = tf.reshape(simCube12,(len0,len1))
			simCube13 = tf.ones([len0,len1])

			simCube = tf.concat((simCube9,simCube5,simCube1,simCube10,simCube6,simCube2,simCube12,simCube8,simCube4,simCube11,simCube7,simCube3,simCube13),0)
			# simCube=torch.unsqueeze(simCube,0)
			# simCube = F.pad(simCube, (0, self.limit - simCube.size(3), 0, self.limit - simCube.size(2)))[0]
			# print(simCube1)
			# print(simCube)
			# print(simCube8)
			# sys.exit()
			return simCube

	def similarity_focus(self,simCube):
			ssize0 = simCube.get_shape().as_list()[0]
			ssize1 = simCube.get_shape().as_list()[1]
			ssize2 = simCube.get_shape().as_list()[2]

			mask = tf.multiply(tf.ones([ssize0,ssize1,ssize2]),0.1)
			s1tag = tf.zeros([ssize1])
			s2tag = tf.zeros([ssize2])
			sorted, indices = tf.nn.top_k(tf.reshape(simCube[6],(-1)), k=ssize1 * ssize2, sorted=True)
			record = []
			for indix in indices:
				pos1 = indix/ssize2
				pos2 = (indix - ssize2*pos1)
				if s1tag[pos1] + s2tag[pos2] <= 0:
					s1tag[pos1] = 1
					s2tag[pos2] = 1
					record.append((pos1,pos2))
					mask[0][pos1][pos2] = mask[0][pos1][pos2] + 0.9
					mask[1][pos1][pos2] = mask[1][pos1][pos2] + 0.9
					mask[2][pos1][pos2] = mask[2][pos1][pos2] + 0.9
					mask[3][pos1][pos2] = mask[3][pos1][pos2] + 0.9
					mask[4][pos1][pos2] = mask[4][pos1][pos2] + 0.9
					mask[5][pos1][pos2] = mask[5][pos1][pos2] + 0.9
					mask[6][pos1][pos2] = mask[6][pos1][pos2] + 0.9
					mask[7][pos1][pos2] = mask[7][pos1][pos2] + 0.9
					mask[8][pos1][pos2] = mask[8][pos1][pos2] + 0.9
					mask[9][pos1][pos2] = mask[9][pos1][pos2] + 0.9
					mask[10][pos1][pos2] = mask[10][pos1][pos2] + 0.9
					mask[11][pos1][pos2] = mask[11][pos1][pos2] + 0.9
				mask[12][pos1][pos2] = mask[12][pos1][pos2] + 0.9
			s1tag = np.zeros([ssize1])
			s2tag = np.zeros([ssize2])
			sorted, indices = tf.nn.top_k(tf.reshape(simCube[7],(-1)), k=ssize1 * ssize2, sorted=True)
			counter = 0
			for indix in indices:
				pos1 = indix/ssize2
				pos2 = (indix - ssize2*pos1)
				if s1tag[pos1] + s2tag[pos2] <= 0:
					counter += 1
					if (pos1,pos2) in record:
						continue
					else:
						s1tag[pos1] = 1
						s2tag[pos2] = 1
						#record.append((pos1,pos2))
						mask[0][pos1][pos2] = mask[0][pos1][pos2] + 0.9
						mask[1][pos1][pos2] = mask[1][pos1][pos2] + 0.9
						mask[2][pos1][pos2] = mask[2][pos1][pos2] + 0.9
						mask[3][pos1][pos2] = mask[3][pos1][pos2] + 0.9
						mask[4][pos1][pos2] = mask[4][pos1][pos2] + 0.9
						mask[5][pos1][pos2] = mask[5][pos1][pos2] + 0.9
						mask[6][pos1][pos2] = mask[6][pos1][pos2] + 0.9
						mask[7][pos1][pos2] = mask[7][pos1][pos2] + 0.9
						mask[8][pos1][pos2] = mask[8][pos1][pos2] + 0.9
						mask[9][pos1][pos2] = mask[9][pos1][pos2] + 0.9
						mask[10][pos1][pos2] = mask[10][pos1][pos2] + 0.9
						mask[11][pos1][pos2] = mask[11][pos1][pos2] + 0.9
				if counter >= len(record):
					break
			focusCube = tf.multiply(simCube,mask)
			return focusCube

	def language_model(self,out0,out1, target_A, target_B,params):
		extra_loss = 0

		size00 = out0.get_shape().as_list()[0]
		size02 = out0.get_shape().as_list()[2]
		size10 = out1.get_shape().as_list()[0]
		size12 = out1.get_shape().as_list()[2]

		h_fw_0, h_bw_0 = self.unpack(tf.reshape(out0,(size00,size02)),half_dim=self.Config.hidden_dim)
		h_fw_1, h_bw_1 = self.unpack(tf.reshape(out1,(size10,size12)),half_dim=self.Config.hidden_dim)
		''''''
		m_fw_0 = tf.tanh(tf.matmul(h_fw_0, params['lm_Wm_forward']))
		m_bw_0 = tf.tanh(tf.matmul(h_bw_0, params['lm_Wm_backward']))
		m_fw_1 = tf.tanh(tf.matmul(h_fw_1, params['lm_Wm_forward']))
		m_bw_1 = tf.tanh(tf.matmul(h_bw_1, params['lm_Wm_backward']))
		q_fw_0 = tf.nn.softmax(tf.matmul(m_fw_0, params['lm_Wq_forward']))
		q_bw_0 = tf.nn.softmax(tf.matmul(m_bw_0, params['lm_Wq_backword']))
		q_fw_1 = tf.nn.softmax(tf.matmul(m_fw_1, params['lm_Wq_forward']))
		q_bw_1 = tf.nn.softmax(tf.matmul(m_bw_1, params['lm_Wq_backword']))

		target_fw_0 = target_A[1:]+[self.tokens.index('</s>')]
		target_bw_0 = [self.tokens.index('<s>')]+target_A[:-1]
		target_fw_1 = target_B[1:]+[self.tokens.index('</s>')]
		target_bw_1 = [self.tokens.index('<s>')]+target_B[:-1]

		loss1 = tf.losses.log_loss(q_fw_0, target_fw_0)
		loss2 = tf.losses.log_loss(q_bw_0, target_bw_0)
		loss3 = tf.losses.log_loss(q_fw_1, target_fw_1)
		loss4 = tf.losses.log_loss(q_bw_1, target_bw_1)

		extra_loss = loss1 + loss2 + loss3 + loss4
		''''''
		return extra_loss


	def deep_cnn(self,focusCube):
			padding = tf.constant([[0,0],[0, self.limit - focusCube.size(2)][0, self.limit - focusCube.size(1)]])
			focusCube = tf.pad(focusCube, padding,mode='constant')

			out = self.layer[0](focusCube)
			out = self.layer[1](out)
			out = self.layer[2](out)
			if self.limit == 16:
				out = self.layer[4](out)
			elif self.limit == 32:
				out = self.layer[3](out)
				out = self.layer[4](out)
			elif self.limit == 48:
				out = self.layer[3](out)
				out = self.layer[5](out)
				#out=self.layer5_1(out)
			#print('debug 6: (out size)')
			#print(out.size())
			out = tf.reshape(out,out.shape.as_list()[0], -1)
			out = self.fc1(out)
			out = self.fc2(out)
			out = tf.substract(tf.multiply(out,2),1)
			#print(out)
			return out

	def mlp(self,focusCube):
		padding = tf.constant([[0,0],[0, self.limit - focusCube.size(3)][0, self.limit - focusCube.size(2)]])
		focusCube = tf.pad(focusCube, padding,mode='constant')
		#print(focusCube.view(-1))
		result = self.mlp_layer(tf.reshape(focusCube,[-1]))
		result = tf.substract(tf.multiply(result,2),1)
		#sys.exit()
		return result

	

	def word_layer(self, lsents, rsents):
		glove_mode = self.Config.word_mode[0]
		update_inv_mode = self.Config.word_mode[1]
		update_oov_mode = self.Config.word_mode[2]

		if glove_mode is True and update_inv_mode is False and update_oov_mode is False:
			try:
				sentA = tf.concat([tf.expand_dims(tf.cast(self.true_dict[word],tf.float32)) for word in lsents], 0)
				# sentA = tf.Variable(sentA)  # .cuda()
				sentB = tf.concat([tf.expand_dims(tf.cast(self.true_dict[word],tf.float32)) for word in rsents], 0)
				# sentB = tf.Variable(sentB)  # .cuda()
			except:
				print(lsents)
				print(rsents)
				sys.exit()
		'''
		elif glove_mode == True and update_inv_mode == False and update_oov_mode == True:
			firstFlag=True
			for word in lsents:
				if firstFlag:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice=indice.cuda()
						output=self.word_embedding(indice)
						firstFlag=False
					else:
						output = Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output=output.cuda()
						firstFlag=False
				else:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new), 0)
					else:
						output_new=Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new=output_new.cuda()
						output = torch.cat((output, output_new), 0)

			sentA = output
			firstFlag = False
			for word in rsents:
				if firstFlag:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
					else:
						output = Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						firstFlag = False
				else:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new), 0)
					else:
						output_new = Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new), 0)
			sentB = output
		elif glove_mode==True and update_inv_mode==True and update_oov_mode==False:
			firstFlag = True
			for word in lsents:
				if firstFlag:
					if word in self.oov:
						output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						output=output.view(1,-1)
						firstFlag = False
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.copied_word_embedding(indice)
						firstFlag = False
				else:
					if word in self.oov:
						output_new=Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new.view(1,-1)), 0)
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.copied_word_embedding(indice)
						output = torch.cat((output, output_new), 0)
			sentA = output
			firstFlag = True
			for word in rsents:
				if firstFlag:
					if word in self.oov:
						output = Variable(
							torch.Tensor([random.uniform(-0.05, 0.05) for i in range(self.embedding_dim)]))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.copied_word_embedding(indice)
						firstFlag = False
				else:
					if word in self.oov:
						output_new = Variable(
							torch.Tensor([random.uniform(-0.05, 0.05) for i in range(self.embedding_dim)]))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new.view(1,-1)), 0)
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.copied_word_embedding(indice)
						output = torch.cat((output, output_new), 0)
			sentB = output
		elif glove_mode==True and update_inv_mode==True and update_oov_mode==True:
			tmp=[]
			for word in lsents:
				try:
					tmp.append(self.word2id[word])
				except:
					tmp.append(self.word2id['oov'])
			indices = Variable(torch.LongTensor(tmp))
			if torch.cuda.is_available():
				indices = indices.cuda()
			sentA = self.copied_word_embedding(indices)
			tmp = []
			for word in rsents:
				try:
					tmp.append(self.word2id[word])
				except:
					tmp.append(self.word2id['oov'])
			indices = Variable(torch.LongTensor(tmp))
			if torch.cuda.is_available():
				indices = indices.cuda()
			sentB = self.copied_word_embedding(indices)
		elif glove_mode==False and update_inv_mode==False and update_oov_mode==False:
			firstFlag = True
			for word in lsents:
				if firstFlag:
					output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
					if torch.cuda.is_available():
						output = output.cuda()
					output = output.view(1, -1)
					firstFlag = False
				else:
					output_new = Variable(self.fake_dict[word].view(1, self.embedding_dim))
					if torch.cuda.is_available():
						output_new = output_new.cuda()
					output = torch.cat((output, output_new.view(1, -1)), 0)
			sentA = output
			firstFlag = True
			for word in rsents:
				if firstFlag:
					output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
					if torch.cuda.is_available():
						output = output.cuda()
					output = output.view(1, -1)
					firstFlag = False
				else:
					output_new = Variable(self.fake_dict[word].view(1, self.embedding_dim))
					if torch.cuda.is_available():
						output_new = output_new.cuda()
					output = torch.cat((output, output_new.view(1, -1)), 0)
			sentB = output
		elif glove_mode==False and update_inv_mode==False and update_oov_mode==True:
			firstFlag = True
			for word in lsents:
				if firstFlag:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
					else:
						output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
				else:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new), 0)
					else:
						output_new = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output_new = output_new.view(1,-1)
						output = torch.cat((output, output_new), 0)
			sentA = output
			firstFlag = True
			for word in rsents:
				if firstFlag:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
					else:
						output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
				else:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new), 0)
					else:
						output_new = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output_new = output_new.view(1, -1)
						output = torch.cat((output, output_new), 0)
			sentB = output
		elif glove_mode==False and update_inv_mode==True and update_oov_mode==False:
			firstFlag = True
			for word in lsents:
				if firstFlag:
					if word in self.oov:
						output = Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
				else:
					if word in self.oov:
						output_new = Variable(torch.Tensor(self.dict[word].view(1, self.embedding_dim)))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new.view(1, -1)), 0)
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new.view(1, -1)), 0)
			sentA = output
			firstFlag = True
			for word in rsents:
				if firstFlag:
					if word in self.oov:
						output = Variable(
							torch.Tensor([random.uniform(-0.05, 0.05) for i in range(self.embedding_dim)]))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
				else:
					if word in self.oov:
						output_new = Variable(
							torch.Tensor([random.uniform(-0.05, 0.05) for i in range(self.embedding_dim)]))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new.view(1, -1)), 0)
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new.view(1, -1)), 0)
			sentB = output
		elif glove_mode==False and update_inv_mode==True and update_oov_mode==True:
			indices=Variable(torch.LongTensor([self.tokens.index(word) for word in lsents]))
			#print(indices)
			if torch.cuda.is_available():
				indices=indices.cuda()
			sentA=self.word_embedding(indices)
			indices = Variable(torch.LongTensor([self.tokens.index(word) for word in rsents]))
			#print(indices)
			if torch.cuda.is_available():
				indices = indices.cuda()
			sentB = self.word_embedding(indices)
		sentA = torch.unsqueeze(sentA, 0).view(-1, 1, self.embedding_dim)
		sentB = torch.unsqueeze(sentB, 0).view(-1, 1, self.embedding_dim)
		'''

		return (sentA, sentB)

	def c2w_cell(self,indices,h,c):

		params = self.Config.params
		input = self.c2w_embedding(indices)
		input = tf.reshape(input,(-1,1,50))

		out,(state,_) = self.lstm_c2w(input,(h,c))
		output_char = tf.matmul(tf.reshape(state[0][0][:],(-1, 1)), params['df']) + tf.matmul( 
			tf.reshape(state[1][0][:],(-1, 1)), params['db']) + tf.reshape(params['bias'],(-1, 1))
		output_char = tf.reshape(output_char,(1, -1))
		return output_char

	def charCNN_cell(self,indices):

		input = self.char_cnn_embedding(indices)
		input = tf.expand_dims(input,0)

		out1 = self.charCNN_filter[0](input)
		out2 = self.charCNN_filter[1](input)
		out3 = self.charCNN_filter[2](input)
		out4 = self.charCNN_filter[3](input)
		out5 = self.charCNN_filter[4](input)
		out6 = self.charCNN_filter[5](input)
		out7 = self.charCNN_filter[6](input)

		final_output = tf.concat([tf.squeeze(out1),tf.squeeze(out2),tf.squeeze(out3),tf.squeeze(out4),tf.squeeze(out5),tf.squeeze(out6),tf.squeeze(out7)])
		final_output = tf.reshape(final_output,(1,-1))

		transform_gate = self.transform_gate(final_output)
		final_output = transform_gate * self.char_cnn_mlp(final_output) + (1-transform_gate) * final_output
		return final_output

	def generate_word_indices(self,word):
	
		'''
		if self.task=='hindi':
			#char_gram = syllabifier.orthographic_syllabify(word, 'hi')
			char_gram = list(splitclusters(word))
			indices=[]
			if self.character_ngrams == 1:
				indices=[self.dict_char_ngram[char] for char in char_gram]
			elif self.character_ngrams == 2:
				if self.character_ngrams_overlap:
					if len(char_gram) <= 2:
						indices = [self.dict_char_ngram[word]]
					else:
						for i in range(len(char_gram) - 1):
							indices.append(self.dict_char_ngram[char_gram[i]+char_gram[i+1]])
				else:
					if len(char_gram) <= 2:
						indices = [self.dict_char_ngram[word]]
					else:
						for i in range(0, len(char_gram) - 1, 2):
							indices.append(self.dict_char_ngram[char_gram[i] + char_gram[i + 1]])
						if len(char_gram)%2==1:
							indices.append(self.dict_char_ngram[char_gram[len(char_gram)-1]])
		'''

		indices = []
		if self.character_ngrams == 1:
			for char in word:
				try:
					indices.append(self.dict_char_ngram[char])
				except:
					continue
			#indices = [self.dict_char_ngram[char] for char in word]
		elif self.character_ngrams == 2:
			if self.character_ngrams_overlap:
				if len(word) <= 2:
					try:
						indices = [self.dict_char_ngram[word]]
					except:
						indices = [self.dict_char_ngram[' ']]
				else:
					for i in range(len(word) - 1):
						try:
							indices.append(self.dict_char_ngram[word[i:i + 2]])
						except:
							indices.append(self.dict_char_ngram[' '])
			else:
				if len(word) <= 2:
					indices = [self.dict_char_ngram[word]]
				else:
					for i in range(0, len(word) - 1, 2):
						indices.append(self.dict_char_ngram[word[i:i + 2]])
					if len(word) % 2 == 1:
						indices.append(self.dict_char_ngram[word[len(word) - 1]])

		elif self.character_ngrams == 3:
			if self.character_ngrams_overlap:
				if len(word) <= 3:
					indices = [self.dict_char_ngram[word]]
				else:
					for i in range(len(word) - 2):
						indices.append(self.dict_char_ngram[word[i:i + 3]])
			else:
				if len(word) <= 3:
					indices = [self.dict_char_ngram[word]]
				else:
					for i in range(0, len(word) - 2, 3):
						indices.append(self.dict_char_ngram[word[i:i + 3]])
					if len(word) % 3 == 1:
						indices.append(self.dict_char_ngram[word[len(word) - 1]])
					elif len(word) % 3 == 2:
						indices.append(self.dict_char_ngram[word[len(word) - 2:]])
		return indices

	def c2w_or_cnn_layer(self,lsents, rsents):
		config = self.Config
		h = tf.zeros(2, 1, config.embedding_dim)  # 2 for bidirection
		c = tf.zeros(2, 1, config.embedding_dim)

		firstFlag = True
		for word in lsents:
			indices = self.generate_word_indices(word)

			# padding for the c2w
			if not Config.c2w_mode:
				if len(indices) < 20:
					indices = indices + [0 for i in range(self.charcnn_max_word_length - len(indices))]
				else:
					indices = indices[0:20]
			if firstFlag:
				if Config.c2w_mode:
					output = self.c2w_cell([indices], h, c)
				else:
					output = self.charCNN_cell([indices])
				firstFlag = False
			else:
				if Config.c2w_mode:
					output_new = self.c2w_cell([indices], h, c)
				else:
					output_new = self.charCNN_cell([indices])
				output = tf.concat((output,output_new),0)
		#print(output)
		#sys.exit()

		sentA = output
		firstFlag = True
		for word in rsents:
			# print word
			indices = self.generate_word_indices(word)
			if not self.c2w_mode:
				if len(indices) < 20:
					indices = indices + [0 for i in range(self.charcnn_max_word_length - len(indices))]
				else:
					indices = indices[0:20]
			# print(indices)
			if firstFlag:
				if self.c2w_mode:
					output = self.c2w_cell([indices], h, c)
				else:
					output = self.charCNN_cell([indices])
				firstFlag = False
			else:
				if self.c2w_mode:
					output_new = self.c2w_cell([indices], h, c)
				else:
					output_new = self.charCNN_cell([indices])
				output = tf.concat((output, output_new), 0)
		#print(output)
		#sys.exit()
		sentB = output
		sentA = tf.reshape(tf.expand_dims(sentA, 0),(-1, 1, config.embedding_dim))
		sentB = tf.reshape(tf.expand_dims(sentB, 0),(-1, 1, config.embedding_dim))
		return (sentA,sentB)

	def mix_cell(self,word, output_word, output_char):
		result = None
		extra_loss = 0
		config = self.Config
		params = self.Config.params
		indices_reduce_dim = [i * 2 for i in range(config.embedding_dim)]

		if self.combine_mode == 'concat':
			result = tf.concat((output_word, output_char), 1)
			result = tf.index_select(result, 1, indices_reduce_dim)
		elif self.combine_mode == 'g_0.25':
			result = 0.25 * output_word + 0.75 * output_char
		elif self.combine_mode == 'g_0.50':
			result = 0.5 * output_word + 0.5 * output_char
		elif self.combine_mode == 'g_0.75':
			result = 0.75 * output_word + 0.25 * output_char
		elif self.combine_mode == 'adaptive':
			gate = tf.sigmoid(tf.matmul(output_word, params['vg']) + params['bg'])
			gate = gate.expand(1,config.embedding_dim)
			result = (1-gate) * output_word + gate * output_char
		elif self.combine_mode == 'attention':
			gate = tf.sigmoid(tf.matmaul(tf.tanh(tf.matmul(output_word, params['W1']) + 
				tf.matmul(output_char, params['W2'])), params['W3']))
			result = gate*output_word+(1-gate)*output_char
			if word not in self.oov:
				extra_loss += (1 - tf.losses.cosine_distance(output_word,output_char, axis=1, reduction=None))

		elif self.combine_mode == 'backoff':
			if word in self.oov:
				result = output_char
			else:
				result = output_word
		return (result, extra_loss)

	def mix_layer(self,lsents,rsents):
		config = self.Config
		h = tf.zeros(2, 1, config.embedding_dim)  # 2 for bidirection
		c = tf.zeros(2, 1, config.embedding_dim)

		firstFlag = True
		#if (index + 1) % (int(42200 / 4)) == 0:
		#	extra_loss=0
		#else:
		#	extra_loss=self.language_model(index, h,c)
		extra_loss = 0
		# print(lsents)
		# print(rsents)
		# sys.exit()
		for word in lsents:
			indices = self.generate_word_indices(word)
			if self.c2w_mode:
				output_char = self.c2w_cell([indices], h, c)
			else:
				if len(indices) < 20:
					indices = indices + [0 for i in range(self.charcnn_max_word_length - len(indices))]
				else:
					indices = indices[0:20]
				output_char = self.charCNN_cell([indices])
				if self.task == 'sts':
					output_char = self.down_sampling_300(output_char)
				else:
					output_char = self.down_sampling_200(output_char)
			#indice=Variable(torch.LongTensor([self.tokens.index(word)]))
			#if torch.cuda.is_available():
			#	indice=indice.cuda()
			#output_word = self.copied_word_embedding(indice).view(1,-1)
			output_word = tf.reshape(self.true_dict[word],(1,-1))

			if firstFlag:
				output, extra_loss = self.mix_cell(word, output_word, output_char)
				output2 = output_char
				firstFlag = False
			else:
				output_new, extra_loss = self.mix_cell(word, output_word, output_char)
				output_new2 = output_char
				output = tf.concat((output, output_new), 0)
				output2 = tf.concat((output2, output_new2), 0)
		sentA = output
		sentA2 = output2
		firstFlag = True
		for word in rsents:
			indices = self.generate_word_indices(word)
			if self.c2w_mode:
				output_char = self.c2w_cell([indices], h, c)
			else:
				if len(indices) < 20:
					indices = indices + [0 for i in range(self.charcnn_max_word_length - len(indices))]
				else:
					indices = indices[0:20]
				output_char = self.charCNN_cell([indices])
				if self.task == 'sts':
					output_char = self.down_sampling_300(output_char)
				else:
					output_char = self.down_sampling_200(output_char)
			#indice = Variable(torch.LongTensor([self.tokens.index(word)]))
			#if torch.cuda.is_available():
			#	indice = indice.cuda()
			#output_word = self.copied_word_embedding(indice).view(1, -1)
			output_word = tf.reshape(self.true_dict[word],(1,-1))

			if firstFlag:
				output, extra_loss = self.mix_cell(word, output_word, output_char)
				output2 = output_char
				firstFlag = False
			else:
				output_new, extra_loss = self.mix_cell(word, output_word, output_char)
				output_new2 = output_char
				output = tf.concat((output, output_new), 0)
				output2 = tf.concat((output2, output_new2), 0)
		
		sentB = output
		sentB2 = output2

		sentA = tf.reshape(tf.expand_dims(sentA, 0),(-1, 1, config.embedding_dim))  # *2)
		sentB = tf.reshape(tf.expand_dims(sentB, 0),(-1, 1, config.embedding_dim))  # *2)
		sentA2 = tf.reshape(tf.expand_dims(sentA2, 0),(-1, 1, config.embedding_dim))  # *2)
		sentB2 = tf.reshape(tf.expand_dims(sentB2, 0),(-1, 1, config.embedding_dim))  # *2)
		return (sentA, sentA2, sentB, sentB2, extra_loss)


	def build(self):
		input_A = self.inputs[0]
		input_B = self.inputs[1]
		extra_loss1 = 0
		extra_loss2 = 0
		raw_input_A = input_A
		raw_input_B = input_B

		h0 = tf.zeros(self.num_layers * 2, 1, self.Config.hidden_dim)
		c0 = tf.zeros(self.num_layers * 2, 1, self.Config.hidden_dim)

                if self.Config.granularity == 'word':
			input_A, input_B = self.word_layer(input_A, input_B)
		elif self.Config.granularity == 'char':
			input_A, input_B = self.c2w_or_cnn_layer(input_A,input_B)
			if self.Config.lm_mode:
				target_A = []  # [self.tokens.index(word) for word in input_A]
				for word in raw_input_A:
					if self.word_freq[word] >= 4:
						target_A.append(self.tokens.index(word))
					else:
						target_A.append(self.tokens.index('oov'))
				target_B = []  # [self.tokens.index(word) for word in input_B]
				for word in raw_input_B:
					if self.word_freq[word] >= 4:
						target_B.append(self.tokens.index(word))
					else:
						target_B.append(self.tokens.index('oov'))
					lm_out0, _ = self.lm_lstm(input_A, (h0, c0))
					lm_out1, _ = self.lm_lstm(input_B, (h0, c0))
					extra_loss2 = self.language_model(lm_out0, lm_out1, target_A, target_B)

		out0, (state0,_) = self.lstm(input_A, (h0, c0))
		out1, (state1,_) = self.lstm(input_B, (h0, c0))
		simCube, _ = self.pairwise_word_interaction(out0,out1, target_A=None, target_B=None)
		focusCube = self.similarity_focus(simCube)
		if self.deep_CNN:
			output = self.deep_cnn(focusCube)
		else:
			output = self.mlp(focusCube)
		output = tf.reshape(output,(1,2))


		return (output,extra_loss1 + extra_loss2)


		pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str, default='msrp',help='Currently supported tasks: pit, url, msrp')
	parser.add_argument('--granularity', type=str, default='word',help='Currently supported granularities: char and word.')
	parser.add_argument('--pretrained', type=bool, default=True,help='Use pretrained word embedding or not')
	parser.add_argument('--char_ngram', type=int, default=2,help='unigram (1), bigram (2) or trigram (3)')
	parser.add_argument('--char_assemble', type=str, default='cnn',help='Assemble char embedding into word embedding: c2w or cnn')
	parser.add_argument('--language_model', type=bool, default=False,help='Use multi task language model or not')
	parser.add_argument('--deep_CNN', type=bool, default=True,help='use 19 layer CNN or not')

	args = parser.parse_args()

	# arguments
	task = args.task
	granularity = args.granularity
	character_ngrams = args.char_ngram
        deep_CNN = args.deep_CNN

	# other configuration and initializations
	num_epochs = 20
	character_ngrams_overlap = False
        lm_mode = False

	token_list = []
	true_dict = {}
	fake_dict = {}
	dict_char_ngram = {}
	word_freq = {}
	oov = []


	if granularity == 'char':

		# c2w parameters
		if args.language_model:
			lm_mode = True
		else:
			lm_mode = False

		if args.char_assemble == 'c2w':
			c2w_mode = True
		else:
			c2w_mode = False

		if c2w_mode:
			EMBEDDING_DIM = 200
		else:
			EMBEDDING_DIM = 1100

	# load data
	basepath = os.path.dirname(os.path.abspath(__file__))
	if task == 'url' or task == 'msrp':
		num_class = 2
		trainset = readURLdata(basepath + '/data/' + task + '/train/',granularity)
		testset = readURLdata(basepath + '/data/' + task + '/test/', granularity)
	else:
		print('wrong input for the first argument!')
		sys.exit()

	if os.path.exists(basepath + '/data/' + task + '/word_freq.p'):
		word_freq,token_list = cPickle.load(open(basepath + '/data/' + task + '/word_freq.p', "rb"))
	# build the dictionary
	else:
		tokens = set()
		lsents, rsents, labels = trainset
		for sent in lsents:
			for word in sent:
				tokens.add(word)
				try:
					word_freq[word] += 1
				except:
					word_freq[word] = 1
		for sent in rsents:
			for word in sent:
				tokens.add(word)
				try:
					word_freq[word] += 1
				except:
					word_freq[word] = 1

		lsents, rsents, labels = testset
		for sent in lsents:
			for word in sent:
				tokens.add(word)
				try:
					word_freq[word] += 1
				except:
					word_freq[word] = 1
		for sent in rsents:
			for word in sent:
				tokens.add(word)
				try:
					word_freq[word] += 1
				except:
					word_freq[word] = 1

		token_list = list(tokens)
		if granularity == 'char':
			token_list.append('<s>')
			token_list.append('</s>')
		token_list.append('oov')
		cPickle.dump([word_freq,token_list], open(basepath + '/data/' + task + '/word_freq.p', "wb"))

	if granularity == 'char':
		if os.path.exists(basepath + '/data/' + task + '/char_dict.p'):
			dict_char_ngram = cPickle.load(open(basepath + '/data/' + task + '/char_dict.p', "rb"))
		else:
			if character_ngrams == 1:
				# dict_char_ngram = pickle.load(open(base_path+ '/char_dict.p', "rb"))
				dict_char_ngram = set()
				for word in token_list:
					for i in range(len(word)):
						dict_char_ngram.add(word[i])
				ngrams_list = list(dict_char_ngram)
				dict_char_ngram = {}
				count = 0
				for unit in ngrams_list:
					dict_char_ngram[unit] = count
					count += 1

			elif character_ngrams == 2 and character_ngrams_overlap:
				# dict_char_ngram = pickle.load(open(base_path+ '/bigram_dict.p', "rb"))
				dict_char_ngram = set()
				for word in token_list:
					if len(word) <= 2:
						dict_char_ngram.add(word)
					else:
						for i in range(len(word) - 1):
							dict_char_ngram.add(word[i:i + 2])
				ngrams_list = list(dict_char_ngram)
				dict_char_ngram = {}
				count = 0
				for unit in ngrams_list:
					dict_char_ngram[unit] = count
					count += 1
			elif character_ngrams == 2 and not character_ngrams_overlap:
				# dict_char_ngram = pickle.load(open(base_path+ '/bigram_dict_no_overlap.p', "rb"))
				dict_char_ngram = set()
				for word in token_list:
					if len(word) <= 2:
						dict_char_ngram.add(word)
					else:
						for i in range(0, len(word) - 1, 2):
							dict_char_ngram.add(word[i:i + 2])
						if len(word) % 2 == 1:
							dict_char_ngram.add(word[len(word) - 1])
				ngrams_list = list(dict_char_ngram)
				dict_char_ngram = {}
				count = 0
				for unit in ngrams_list:
					dict_char_ngram[unit] = count
					count += 1
			elif character_ngrams == 3 and character_ngrams_overlap:
				# dict_char_ngram = pickle.load(open(base_path+ '/trigram_dict.p', "rb"))
				dict_char_ngram = set()
				for word in token_list:
					if len(word) <= 3:
						dict_char_ngram.add(word)
					else:
						for i in range(len(word) - 2):
							dict_char_ngram.add(word[i:i + 3])
				ngrams_list = list(dict_char_ngram)
				dict_char_ngram = {}
				count = 0
				for unit in ngrams_list:
					dict_char_ngram[unit] = count
					count += 1
			elif character_ngrams == 3 and not character_ngrams_overlap:
				# dict_char_ngram = pickle.load(open(base_path+ '/trigram_dict_no_overlap.p', "rb"))
				dict_char_ngram = set()
				for word in token_list:
					if len(word) <= 3:
						dict_char_ngram.add(word)
					else:
						for i in range(0, len(word) - 2, 3):
							dict_char_ngram.add(word[i:i + 3])
						if len(word) % 3 == 1:
							dict_char_ngram.add(word[len(word) - 1])
						elif len(word) % 3 == 2:
							dict_char_ngram.add(word[len(word) - 2:])
				ngrams_list = list(dict_char_ngram)
				dict_char_ngram = {}
				count = 0
				for unit in ngrams_list:
					dict_char_ngram[unit] = count
					count += 1
			dict_char_ngram[' '] = len(dict_char_ngram)
			cPickle.dump(open(dict_char_ngram, basepath + '/data/' + task + '/char_dict.p', "wb"))


		# word_freq = pickle.load(open(basepath + '/data/' + task + '/word_freq.p', "rb"))

		print('current task: ' + task + ', lm mode: ' + str(lm_mode) + ', c2w mode: ' + str(c2w_mode) + ', n = ' + str(
			character_ngrams) + ', overlap = ' + str(character_ngrams_overlap) + '.')

	elif granularity == 'word':

		num_inv = 0
		num_oov = 0
		if args.pretrained:
			glove_mode = True
		else:
			glove_mode = False

		update_inv_mode = False
		update_oov_mode = False
		word_mode = (glove_mode, update_inv_mode, update_oov_mode)

		num_oov = 0
		num_inv = 0

		true_dict, fake_dict, oov = cPickle.load(open(basepath + '/data/' + task + '_emb.p', "rb"))

		if task == 'msrp':
			EMBEDDING_DIM = 300

		elif task == 'url' or task == 'pit':
			EMBEDDING_DIM = 200			


		print('finished loading word vector, there are ' + str(num_inv) + ' INV words and ' + str(
			num_oov) + ' OOV words.')
		print('current task: ' + task + ', glove mode = ' + str(glove_mode) + ', update_inv_mode = ' + str(
			update_inv_mode) + ', update_oov_mode = ' + str(update_oov_mode))

	else:
		print('wrong input for the second argument!')
		sys.exit()

	config = Config(granularity,deep_CNN,word_mode,dict_char_ngram,word_freq,oov,token_list,lm_mode,EMBEDDING_DIM)


	# build graph and run 
	GPU_config = tf.ConfigProto()
	GPU_config.gpu_options.per_process_gpu_memory_fraction = 0.2
	with tf.Graph().as_default():
		tf.set_random_seed(1234)
		logger.info("Building model")
		start = time.time()

		model = DeepPairWiseWord(config,true_dict,fake_dict)
	
		logger.info("time to build the model: %d", time.time() - start)
		logger.info("the output path: %s", model.Config.output_path)
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		'''
		if not os.path.exists(model.Config.output_path):
			os.makedirs(model.Config.output_path)

		if not os.path.exists(model.Config.output_path_results):
			os.makedirs(model.Config.output_path_results)
		'''

		with tf.Session(config=GPU_config) as session:
			path = ''
			session.run(init)
			lsents, rsents, labels = trainset
			labels = tf.one_hot(labels)

			max_result = -1
			batch_size = 32
			report_interval = 50000

			path = model.Config.model_path
			optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
			tvs = tf.trainable_variables()
			for epoch in range(Config.max_epochs):

				logger.info("running epoch %d", epoch)
				indices = np.random_permutation(len(lsents))
				data_loss = 0
				train_correct = 0
				accum_grads = [tf.zeros_like(tv.initial_value()) for tv in tvs]

				for index,i in enumerate(indices):
					sentA = lsents[i]
					sentB = rsents[i]
					output,extra_loss = model(sentA, sentB)
					output = tf.devide(output + 1,2)

					loss = tf.losses.hinge_loss(output,labels) + extra_loss
					grad_var = optimizer.compute_gradients(loss)
					accum_grads = [accum_grads[i] + gv[0] for i,gv in enumerate(grad_var)]

					curr_out, curr_loss, _ = session.run([output,loss,accum_grads],{model.inputs:[sentA,sentB]})
					data_loss = data_loss + curr_loss
					if np.argmax(labels[i]) == np.argmax(curr_out):
						train_correct += 1

					if (index+1) % batch_size == 0:
						train_op = optimizer.apply_gradients([(accum_grads[i],gv[1]) for i,gv in enumerate(grad_var)])
						session.run(train_op)
						accum_grad = [tf.zeros_like(accum_grad) for accum_grad in accum_grads]

					if (index+1) % report_interval == 0:
						msg = '%d completed epochs, %d batches' % (epoch, index+1)
						msg += '\t train batch loss: %f' % (data_loss / (index+1))
						train_acc = train_correct / (index + 1)
						print(msg)

					if (index + 1) % (int(len(lsents)/2)) == 0:


						# test on URL dataset
						#print('testing on URL dataset:')
						#testset = readURLdata(basepath + '/data/url/test_9324/', granularity)
						test_lsents, test_rsents, test_labels = testset
						predicted = []
						gold = []
						correct = 0
						for test_i in range(len(test_lsents)):
							sentA = test_lsents[test_i]
							sentB = test_rsents[test_i]
							output, _ = model(sentA, sentB)
							output = tf.devide(output + 1,2)

							curr_out = session.run(output,{model.inputs:[sentA,sentB]})
							predicted.append(curr_out[1])
							gold.append(test_labels[test_i])

						_ , result = URL_maxF1_eval(predict_result=predicted, test_data_label=gold)
						if result > max_result:
							max_result = result
						elapsed_time = time.time() - start

						print('Epoch ' + str(epoch + 1) + ' finished within ' + str(timedelta(seconds=elapsed_time))+', and current time:'+ str(datetime.now()))
						print('Best result until now: %.6f' % max_result)

					

			#	cPickle.dump(cnn_encodings,open("./result_lm/{:%Y%m%d_%H%M%S}/".format(datetime.now()) + 'lb_emd_' + args.label_freq + '.p',"wb"))


