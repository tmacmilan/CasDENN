import numpy as np
import six.moves.cPickle as pickle
#from model import config
import config_APS_1 as config
import networkx as nx
import scipy.sparse
import gc
import math
LABEL_NUM = 0
n_time_interval= config.n_time_interval
time_interval = config.time_interval
degree_interval_list= config.degree_interval_list
n_degree_interval=len(degree_interval_list)+1
# trans the original ids to 1~n
class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i, cnt)
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)

    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]

    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]

    def length(self):
        return len(self.new_to_original)

#trainsform the sequence to list
def sequence2list(flename):
    graphs = {}
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = [] #walk[0] = cascadeID
            for i in range(1, len(walks)):
                s = walks[i].split(":")[0] #node
                t = walks[i].split(":")[1] #time
                graphs[walks[0]].append([[str(xx) for xx in s.split(",")],int(t)])
    return graphs

#read label and size from cascade file
def read_labelANDsize(filename):
    labels = {}
    sizes = {}
    with open(filename, 'r') as f:
        for line in f:
            profile = line.split('\t')
            labels[profile[0]] = profile[-1]
            sizes[profile[0]] = int(profile[3])
    return labels,sizes

def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            for i in walk[0]:
                original_ids.add(i)
    print ("length of original isd:",len(original_ids))
    return original_ids

def get_nodes(graph):
    nodes = {}
    j = 0
    for walk in graph:
        for i in walk[0]:
            if i not in nodes.keys():
                nodes[i] = j
                j = j+1
    return nodes
def project(degree_dis_Tmp,degree_interval_list):#将degree_dis_Tmp映射到degree_interval_list区间内
    degree_his=np.zeros(shape=(1, len(degree_interval_list)+1))
    j_begin = 0
    for k in range(len(degree_interval_list)):
        for j in range (j_begin,len(degree_dis_Tmp)):
            if j<= degree_interval_list[k]:
                degree_his[0,k] += degree_dis_Tmp[j]
                j_begin +=1
            else:
                break
    if j_begin<len(degree_dis_Tmp):
        for j in range (j_begin,len(degree_dis_Tmp)):
            degree_his[0,-1] += degree_dis_Tmp[j]
    return degree_his
        
        
def write_XYSIZE_data(graphs,labels,sizes,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num, filename):
    #get the x,y,and size  data
    id_data = []
    x_data = []
    y_data = []
    sz_data = []
    time_data = []
    Laplacian_data = []
    node_sizeList =[]
    edge_siezeList = []
    for key,graph in graphs.items():
        id = key
        label = labels[key].split()
        y = int(label[LABEL_NUM]) #label
        temp_time = [] #store time
        size_temp = len(graph)
        if size_temp !=  sizes[key]:
            print (size_temp,sizes[key])
        nodes_items = get_nodes(graph)
        nodes_list = nodes_items.values()
        nx_G = nx.DiGraph()
        nx_G.add_nodes_from(nodes_list)
        graph = sorted(graph,key=(lambda x:x[1]))## 没有按时间排序？
        tmp_degree=np.zeros(shape=(n_time_interval, n_degree_interval))
        k=0
        for walk in graph:
            walk_time = walk[1] ## 没有按时间排序？
            #print (walk_time)
            temp_time.append(walk_time)
            k_new = int(math.floor(walk_time/ time_interval))
            if k_new > k:## degree distribution
                degree_dis_Tmp=nx.degree_histogram(nx_G)
                tmp_degree[k]= project(degree_dis_Tmp,degree_interval_list)#转换成 n_degree_interval个数
            k = k_new
            if walk_time == 0:
                nx_G.add_edge(nodes_items.get(walk[0][0]), nodes_items.get(walk[0][0]))
            for i in range(len(walk[0])-1):
                nx_G.add_edge(nodes_items.get(walk[0][i]),nodes_items.get(walk[0][i+1]))
        for kth in range(k,n_time_interval):
            degree_dis_Tmp = nx.degree_histogram(nx_G)
            tmp_degree[k]= project(degree_dis_Tmp,degree_interval_list)        
        #caculate laplacian
        node_sizeList.append(len(nx_G.nodes()))
        edge_siezeList.append(len(nx_G.edges()))
        Laplacian=[]
        time_data.append(temp_time)
        id_data.append(id)
        x_data.append(tmp_degree)
        y_data.append(np.log(y+1.0)/np.log(2.0))
        Laplacian_data.append(Laplacian)
        sz_data.append(size_temp)
    gc.collect()
    print ('size:'+str(len(labels))+'\n')
    print ('ave.node.size:'+str(sum(node_sizeList)/len(node_sizeList))+'\n')
    print ('ave.edge.size:'+str(sum(edge_siezeList)/len(edge_siezeList)))
    pickle.dump((id_data,x_data,Laplacian_data,y_data, sz_data, time_data,index.length()), open(filename,'wb'))

def get_maxsize(sizes):
    max_size = 0
    for cascadeID in sizes:
        max_size = max(max_size,sizes[cascadeID])
    gc.collect()
    return max_size

def get_max_length(graphs):
    len_sequence = 0
    max_num = 0
    for cascadeID in graphs:
        max_num = max(max_num,len(graphs[cascadeID]))
        for sequence in graphs[cascadeID]:
            len_sequence = max(len_sequence,len(sequence[0]))
    gc.collect()
    return len_sequence

def get_max_node_num(graphs):
    max_num = 0
    for key,graph in graphs.items():
        nodes = get_nodes(graph)
        max_num = max(max_num,len(nodes))
    return max_num
if __name__ == "__main__":

    ### data set ###
    graphs_train = sequence2list(config.shortestpath_train)
    graphs_val = sequence2list(config.shortestpath_val)
    graphs_test = sequence2list(config.shortestpath_test)

    ## get Laplacian ##
    cascade_train = config.cascade_train
    cascade_test = config.cascade_test
    cascade_val = config.cascade_val

    ### get labels ###
    labels_train, sizes_train = read_labelANDsize(config.cascade_train)  # 标签以及观测到的长度
    labels_val, sizes_val = read_labelANDsize(config.cascade_val)
    labels_test, sizes_test = read_labelANDsize(config.cascade_test)
    NUM_SEQUENCE = max(get_maxsize(sizes_train),get_maxsize(sizes_val),get_maxsize(sizes_test))# 观测到的最大长度

    LEN_SEQUENCE_train = get_max_length(graphs_train) #最大步长
    LEN_SEQUENCE_val = get_max_length(graphs_val)
    LEN_SEQUENCE_test = get_max_length(graphs_test)
    LEN_SEQUENCE = max(LEN_SEQUENCE_train,LEN_SEQUENCE_val,LEN_SEQUENCE_test)

    max_num_train = get_max_node_num(graphs_train)
    max_num_test = get_max_node_num(graphs_test)
    max_num_val = get_max_node_num(graphs_val)
    max_num = max(max_num_train, max_num_test, max_num_val)

    # get the total original_ids and tranform the index from 0 ~n-1
    original_ids = get_original_ids(graphs_train)\
                    .union(get_original_ids(graphs_val))\
                    .union(get_original_ids(graphs_test))

    original_ids.add(-1)
    ## index is new index
    index = IndexDict(original_ids)

    print("create train")
    write_XYSIZE_data(graphs_train, labels_train,sizes_train,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num, config.train_pkl)
    print("create val an test")
    write_XYSIZE_data(graphs_val, labels_val, sizes_val,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num, config.val_pkl)
    write_XYSIZE_data(graphs_test, labels_test, sizes_test,LEN_SEQUENCE,NUM_SEQUENCE,index,max_num,config.test_pkl)
    pickle.dump((len(original_ids),NUM_SEQUENCE,LEN_SEQUENCE), open(config.information,'wb'))
    print("Finish!!!")

