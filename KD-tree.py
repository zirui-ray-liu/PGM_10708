import numpy as np
class TreeNode:
    def __init__(self, point):
        self.left = None
        self.right = None
        self.point = point

class KDTree:
    def __init__(self, Points):
        names = self.__dict__
        self.points = Points
        self.n = len(self.points)
        self.k = len(self.points[0])
        self.coordinates = {}
        self.indices = {}
        for i in range(self.k):
            self.coordinates.update({i: self.points[:,i]})
        indices = {}
        for i in range(self.k):
            names['sort_x%s' % i] = self.mergeSort_by_SuperKey(list(np.arange(self.n)), i)
            indices.update({i: names['sort_x%s' % i]})
        self.root = self.buildKdTree(indices, 0)

    def buildKdTree(self, indices, depth):
        d = depth % self.k
        n = len(indices[d])
        if n == 0:
            return None
        root = TreeNode(self.points[indices[d][n//2]])
        indices1, indices2 = {d: indices[d][0:n//2]}, {d: indices[d][n//2+1:n]}
        for i in range(1, self.k):
            j = (d + i) % self.k
            tmp1, tmp2 = [], []
            for index in indices[j]:
                if index == indices[d][n//2]:
                    continue
                if index in indices1[d]:
                    tmp1.append(index)
                if index in indices2[d]:
                    tmp2.append(index)
            indices1.update({j : tmp1})
            indices2.update({j : tmp2})

        root.left = self.buildKdTree(indices1, depth+1)
        root.right = self.buildKdTree(indices2, depth+1)
        return root

    def search_nearest(self, query_point):
        root, futher_node, k = self.root, None, 0
        stack = []
        while root:
            if query_point[k] <= root.point[k]:
                futher_node = root.right
                stack.append((root, futher_node, k))
                root = root.left
            elif query_point[k] > root.point[k]:
                futher_node = root.left
                stack.append((root, futher_node, k))
                root = root.right
            k = (k + 1) % self.k

        min_dist = float("inf")
        nearest_node = None
        while len(stack) != 0:
            current_node, curr_futher_node, current_k = stack.pop()
            dist = self.Squared_Euclid_Distance(current_node, query_point)
            if dist < min_dist:
                min_dist = dist
                nearest_node = current_node
            if min_dist > self.Squared_Point_To_Plane_Distance(current_node, query_point, current_k):
                root, futher_node, k = curr_futher_node, None, current_k
                while root:
                    if query_point[k] <= root.point[k]:
                        futher_node = root.right
                        stack.append((root, futher_node, k))
                        root = root.left
                    elif query_point[k] > root.point[k]:
                        futher_node = root.left
                        stack.append((root, futher_node, k))
                        root = root.right
                    k = (k + 1) % self.k

        return nearest_node.point

    def mergeSort_by_SuperKey(self, index_set, d):
        if len(index_set) == 1:
            return index_set
        l1 = self.mergeSort_by_SuperKey(index_set[0:len(index_set)//2], d)
        l2 = self.mergeSort_by_SuperKey(index_set[len(index_set)//2:len(index_set)], d)
        return self.merge(l1, l2, d)

    def merge(self, l1, l2, d):
        i, j = 0, 0
        res = []
        while(i < len(l1) and j < len(l2)):
            if self.coordinates[d][l1[i]] < self.coordinates[d][l2[j]] :
                res.append(l1[i])
                i += 1
            elif self.coordinates[d][l1[i]] > self.coordinates[d][l2[j]]:
                res.append(l2[j])
                j += 1
            else:
                next_d = (d + 1) % self.k
                while self.coordinates[next_d][l1[i]] == self.coordinates[next_d][l2[j]]:
                    next_d = (next_d + 1) % self.k
                if self.coordinates[next_d][l1[i]] < self.coordinates[next_d][l2[j]]:
                    res.append(l1[i])
                    i += 1
                else:
                    res.append(l2[j])
                    j += 1
        if i != len(l1):
            res.extend(l1[i:len(l1)])
        if j != len(l2):
            res.extend(l2[j:len(l2)])
        return res

    def Squared_Euclid_Distance(self, node, query_point):
        res = 0
        for i in range(len(node.point)):
            res += (node.point[i] - query_point[i])**2
        return res

    def Squared_Point_To_Plane_Distance(self, node, query_point, i):
        return (node.point[i] - query_point[i])**2

    def inorder_traversal(self, node):
        if node == None:
            return
        self.inorder_traversal(node.left)
        print(node.point)
        self.inorder_traversal(node.right)

if __name__ == "__main__":
    # Test_Set = np.array([[2, 3, 3], [5, 4, 2], [9, 6, 7], [4, 7, 9], [8, 1, 5], [7, 2, 6], [9, 4, 1], [8, 4, 2],
    #                      [9, 7, 8], [6, 3, 1], [3, 4, 5], [1, 6, 8], [9, 5, 3], [2, 1, 3], [8, 7, 6]])
    Test_Set = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    kdt = KDTree(Test_Set)
    nearest_point = kdt.search_nearest([5, 4.5])
    #print(nearest_point)
    #kdt.inorder_traversal(kdt.root)