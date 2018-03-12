

class Cluster(object):
    def __init__(self, feature_path, label_path=None):
        self.MAX_LENGTH = 50.
        self.MAX_Z = 50.
        self.feature_path = feature_path
        self.label_path = label_path

        self.label = None
        self.length = None
        self.z = None
        self.features = []

        f = open(self.feature_path, 'r')
        ind = 0
        for line in f:
            if line != '\n':
                if ind == 0:
                    self.label = int(line)
                elif ind == 1:
                    self.length = float(line) / self.MAX_LENGTH
                elif ind == 2:
                    self.z = float(line) / self.MAX_Z
                else:
                    self.features.append(float(line))
                ind = ind + 1
        f.close()
