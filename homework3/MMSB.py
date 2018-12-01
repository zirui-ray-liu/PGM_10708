import numpy as np
class MMSB:
    def __init__(self):
        with open('./hw3train.data') as data:
            n = 0
            self.vertexes = {}
            for lines in data.readlines()[1:]:
                line = lines.strip()
                fields = line.split(" ")
                fields = map(lambda x: int(x), fields)
                if n not in self.vertexes.keys():
                    vertex = Vertex(n)
                    self.vertexes[n] = vertex
                for ID in fields:
                    if ID not in self.vertexes.keys():
                        vertex_ = Vertex(ID)
                        self.vertexes[ID] = vertex_
                    self.vertexes[n].add_neibour(ID, self.vertexes[ID])
                n += 1

        self.num_of_vertex = n
        self.eta0 = 0.01
        self.eta1 = 0.05
        self.alpha = 0.02
        self.topic_size = 5
        self.community_map = np.random.randint(0, self.topic_size, [n, n])
        self.community_map -= np.diag(np.diagonal(self.community_map + 1))
        self.linked_map = np.zeros_like(self.community_map, dtype=int)
        for i in self.vertexes.keys():
            for j in self.vertexes[i].neibours.keys():
                self.linked_map[i][j] = 1
        self.linked_map -= np.diag(np.diagonal(self.linked_map + 1))
        self.community = {}
        for ith_vertex in range(self.num_of_vertex):
            for jth_vertex in range(self.num_of_vertex):
                if ith_vertex == jth_vertex: continue
                k = self.community_map[ith_vertex][jth_vertex]
                if k not in self.community.keys(): self.community[k] = {}
                self.community[k].update({(ith_vertex,jth_vertex): True})


    def Gibbs_Sampling(self, iter=10000):
        for i in range(iter):
            # Update Z (community_map)
            for ith_vertex in range(self.num_of_vertex):
                for jth_vertex in range(ith_vertex+1, self.num_of_vertex):
                    if jth_vertex == ith_vertex: continue
                    print(ith_vertex, jth_vertex)
                    prob_matrix = np.zeros([self.topic_size, self.topic_size], dtype=np.float)
                    for ki in range(self.topic_size):
                        for kj in range(self.topic_size):
                            n_i_ki = np.sum(self.community_map[ith_vertex] == ki) - int(self.community_map[ith_vertex][jth_vertex] == ki)
                            n_j_kj = np.sum(self.community_map[jth_vertex] == kj) - int(self.community_map[jth_vertex][ith_vertex] == kj)
                            n_ki_kj_0, n_ki_kj_1, n_ki_kj = self.cal_n_ki_kj(ki,kj,ith_vertex,jth_vertex)
                            if self.linked_map[ith_vertex][jth_vertex] == 1:
                                prob_matrix[ki][kj] = (n_i_ki + self.alpha)*(n_j_kj + self.alpha)*(n_ki_kj_1+self.eta1)/(n_ki_kj+self.eta1+self.eta0)
                            elif self.linked_map[ith_vertex][jth_vertex] == 0:
                                prob_matrix[ki][kj] = (n_i_ki + self.alpha)*(n_j_kj + self.alpha)*(n_ki_kj_0 + self.eta0)/(n_ki_kj + self.eta1 + self.eta0)
                    prob_matrix/=np.sum(prob_matrix)
                    self.community_map[ith_vertex][jth_vertex], self.community_map[jth_vertex][ith_vertex] = self.get_random_Z_samples(prob_matrix)
            # Update theta



    def cal_n_ki_kj(self,ki, kj, excluded_i,excluded_j):
        n_ki_kj_0, n_ki_kj_1 = 0, 0
        for item in self.community[ki].keys():
            if excluded_i == item[0] and excluded_j == item[1]: continue
            item_T = (item[1], item[0])
            if self.community[kj].get(item_T) and self.linked_map[item[0], item[1]] == 1:
                n_ki_kj_1 += 1
            elif self.community[kj].get(item_T) and self.linked_map[item[0],item[1]] == 0:
                n_ki_kj_0 += 1
        return n_ki_kj_0, n_ki_kj_1, n_ki_kj_1+n_ki_kj_0


    def get_random_Z_samples(self, prob_matrix_):
        prob_matrix_ = np.reshape(prob_matrix_,-1)
        n = np.argmax(np.random.multinomial(20, prob_matrix_))
        return n//self.topic_size, n % self.topic_size

class Vertex:
    def __init__(self, id):
        """
        :param name: vertex's id
        :param childrens: vertex's childrens, type:list
        """
        self.id = id
        self.neibours = {}
    def add_neibour(self, neibour_id, neibour_):
        self.neibours[neibour_id] = neibour_

if __name__ == "__main__":
    mmsb = MMSB()
    mmsb.Gibbs_Sampling()
