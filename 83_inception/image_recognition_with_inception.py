import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class Preprocessing:
    def __init__(self):
        self.__id_map_proto = "./inception_model/imagenet_2012_challenge_label_map_proto.pbtxt"
        self.__proto_map_human = "./inception_model/imagenet_synset_to_human_label_map.txt"
        self.__id_map_proto_dict = dict()
        self.__proto_map_human_dict = dict()
        self.id_map_human = dict()
        
    def preprocessing(self):
        contents_0 = tf.gfile.GFile(self.__proto_map_human).readlines()
        for line in contents_0:
            line = line.strip().split("\t")
            self.__proto_map_human_dict[line[0]] = line[1]
        contents_1 = tf.gfile.GFile(self.__id_map_proto).readlines()
        for line in contents_1:
            if line.startswith("  target_class:"):
                c_id = line.strip().split(": ")[1]
            if line.startswith("  target_class_string:"):
                proto = line.strip().split(": ")[1][1:-1]
                self.__id_map_proto_dict[int(c_id)] = proto
            
        for c_id, proto in self.__id_map_proto_dict.items():
            self.id_map_human[c_id] = self.__proto_map_human_dict[proto]
        return self.id_map_human
        

def classification_by_inception(img_path):
    with tf.gfile.GFile("./inception_model/classify_image_graph_def.pb", "rb") as f:
        default_graph = tf.GraphDef()
        default_graph.ParseFromString(f.read())
        tf.import_graph_def(default_graph, name="")  # 创建图存放已经训练好的inception模型
        
    with tf.Session() as sess:
        softmax_op = sess.graph.get_tensor_by_name("softmax:0")
        img = tf.gfile.FastGFile(img_path, "rb").read()
        prediction = np.squeeze(sess.run(softmax_op, feed_dict={"DecodeJpeg/contents:0": img}))
        
        top_5 = np.argsort(prediction)[-1:-6:-1]
        result = []
        for index in top_5:
            result.append([id_map_human[index], prediction[index]])
        return result
        
        
def application():
    img_path = input("input image path and name:")
    img = Image.open(img_path)
    result = classification_by_inception(img_path)
    
    grid = plt.GridSpec(4, 7)
    plt.subplot(grid[:, :4])
    plt.axis("off")
    plt.imshow(img)
    plt.subplot(grid[1:3, 5:])
    bar_name, bar_value = [], []
    for item in result:
        print("prob %.8f ------ %s" % (item[1], item[0]))
        bar_name.append(item[0].replace(", ", "\n"))
        bar_value.append(item[1])
    plt.yticks(fontsize=7)
    plt.barh(range(len(bar_value)), bar_value, height=0.9, tick_label=bar_name)
    for x, y in zip(range(len(bar_value)), bar_value):
        plt.text(y + 0.005, x - 0.005, y, fontsize=7)
    file_name = str(img_path.split(".")[-2].split("/")[-1]) + "_inception.jpg"
    plt.savefig("./result/" + file_name)
    
    
if __name__ == "__main__":
    pro = Preprocessing()
    id_map_human = pro.preprocessing()
    application()