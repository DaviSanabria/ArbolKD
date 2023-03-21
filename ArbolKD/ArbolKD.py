import os
from sklearn.tree import export_graphviz, _tree
from IPython.display import Image
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from pydotplus import graph_from_dot_data
from PIL import Image as Imagen

# load dataset
dataset = pd.read_csv("page-blocks.csv", header=0, delimiter=';')

# define los atributos
feature_cols = ['height', 'length', 'area', 'eccen', 'pblack', 'pand', 'meantr', 'blackpix' , 'blackand', 'wbtrans']
# define el objetivo
target_variable = ['classification']
#posibles nombres del target
class_names = ['1', '2', '3', '4', '5']
X = dataset[feature_cols]  # Features
Y = dataset[target_variable] # Target variable
# Tree definition
tree = DecisionTreeClassifier()
tree.fit(X,Y)
#visualización del arbol gráfico
dot_data = export_graphviz(tree, feature_names=feature_cols, class_names=class_names)
graph = graph_from_dot_data(dot_data)
graph.write_png('pageblocks.png')
Image(graph.create_png())
# Abre la imagen PNG
imagen = Imagen.open("pageblocks.png")
# Muestra la imagen
imagen.show()
# Exportar el árbol de decisión como texto
tree_rules = export_text(tree, feature_names=feature_cols)
file = open("DaRules.txt", "w")
file.write('DA RULES \n')
file.write(tree_rules)
file.close()

# De las reglas al codigo
#tree_to_code(tree, feature_cols)
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def predict({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node],4)
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            value =  (tree_.value[node]).tolist()
            value = value[0]
            maximum_index = [value.index(max(value))+1, value]
            print ("{}return {}".format(indent, maximum_index))
    recurse(0, 1)

def predict(height, length, area, eccen, pblack, pand, meantr, blackpix, blackand, wbtrans):
  if height <= 3.5:
    if meantr <= 1.355:
      if blackpix <= 44.5:
        if pblack <= 0.45:
          return [1, [49.0, 0.0, 0.0, 0.0, 0.0]]
        else:  # if pblack > 0.45
          return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
      else:  # if blackpix > 44.5
        return [2, [0.0, 4.0, 0.0, 0.0, 0.0]]
    else:  # if meantr > 1.355
      if eccen <= 7.5:
        if eccen <= 2.6665:
          return [1, [20.0, 0.0, 0.0, 0.0, 0.0]]
        else:  # if eccen > 2.6665
          if meantr <= 9.5:
            if pand <= 0.9775:
              if pblack <= 0.3695:
                if area <= 27.5:
                  return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if area > 27.5
                  return [2, [0.0, 2.0, 0.0, 0.0, 0.0]]
              else:  # if pblack > 0.3695
                return [1, [6.0, 0.0, 0.0, 0.0, 0.0]]
            else:  # if pand > 0.9775
              if blackpix <= 7.5:
                return [1, [3.0, 2.0, 0.0, 0.0, 0.0]]
              else:  # if blackpix > 7.5
                return [2, [0.0, 2.0, 0.0, 0.0, 0.0]]
          else:  # if meantr > 9.5
            return [2, [0.0, 2.0, 0.0, 0.0, 0.0]]
      else:  # if eccen > 7.5
        if pblack <= 0.238:
          if eccen <= 22.3335:
            return [1, [7.0, 0.0, 0.0, 0.0, 0.0]]
          else:  # if eccen > 22.3335
            if pblack <= 0.167:
              return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
            else:  # if pblack > 0.167
              return [2, [0.0, 3.0, 0.0, 0.0, 0.0]]
        else:  # if pblack > 0.238
          if blackpix <= 7.5:
            if pblack <= 0.5415:
              if wbtrans <= 2.5:
                return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
              else:  # if wbtrans > 2.5
                return [2, [0.0, 3.0, 0.0, 0.0, 0.0]]
            else:  # if pblack > 0.5415
              if eccen <= 10.5:
                if pblack <= 0.8265:
                  return [1, [1.0, 1.0, 0.0, 0.0, 0.0]]
                else:  # if pblack > 0.8265
                  return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
              else:  # if eccen > 10.5
                return [1, [3.0, 0.0, 0.0, 0.0, 0.0]]
          else:  # if blackpix > 7.5
            if meantr <= 2.86:
              if height <= 1.5:
                if meantr <= 2.5:
                  if blackpix <= 13.5:
                    return [2, [0.0, 14.0, 0.0, 0.0, 0.0]]
                  else:  # if blackpix > 13.5
                    if blackpix <= 14.5:
                      return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if blackpix > 14.5
                      return [2, [0.0, 5.0, 0.0, 0.0, 0.0]]
                else:  # if meantr > 2.5
                  if blackand <= 32.0:
                    return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
                  else:  # if blackand > 32.0
                    return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
              else:  # if height > 1.5
                if eccen <= 8.25:
                  return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                else:  # if eccen > 8.25
                  return [1, [3.0, 0.0, 0.0, 0.0, 0.0]]
            else:  # if meantr > 2.86
              if pblack <= 0.291:
                if height <= 1.5:
                  return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if height > 1.5
                  return [2, [0.0, 2.0, 0.0, 0.0, 0.0]]
              else:  # if pblack > 0.291
                if area <= 27.5:
                  if eccen <= 24.0:
                    if blackpix <= 10.5:
                      if pblack <= 0.7595:
                        return [2, [0.0, 8.0, 0.0, 0.0, 0.0]]
                      else:  # if pblack > 0.7595
                        if pblack <= 0.8445:
                          if length <= 11.5:
                            return [1, [1.0, 1.0, 0.0, 0.0, 0.0]]
                          else:  # if length > 11.5
                            return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                        else:  # if pblack > 0.8445
                          if meantr <= 8.5:
                            return [2, [0.0, 6.0, 0.0, 0.0, 0.0]]
                          else:  # if meantr > 8.5
                            if blackand <= 9.5:
                              return [1, [1.0, 1.0, 0.0, 0.0, 0.0]]
                            else:  # if blackand > 9.5
                              return [2, [0.0, 5.0, 0.0, 0.0, 0.0]]
                    else:  # if blackpix > 10.5
                      return [2, [0.0, 36.0, 0.0, 0.0, 0.0]]
                  else:  # if eccen > 24.0
                    if blackpix <= 25.5:
                      if eccen <= 26.0:
                        return [1, [1.0, 0.0, 0.0, 1.0, 0.0]]
                      else:  # if eccen > 26.0
                        return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if blackpix > 25.5
                      return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                else:  # if area > 27.5
                  if blackpix <= 675.0:
                    if eccen <= 40.5:
                      if blackand <= 192.5:
                        if eccen <= 39.5:
                          return [2, [0.0, 44.0, 0.0, 0.0, 0.0]]
                        else:  # if eccen > 39.5
                          if blackpix <= 35.5:
                            return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                          else:  # if blackpix > 35.5
                            if blackpix <= 38.0:
                              return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                            else:  # if blackpix > 38.0
                              return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                      else:  # if blackand > 192.5
                        return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if eccen > 40.5
                      return [2, [0.0, 139.0, 0.0, 0.0, 0.0]]
                  else:  # if blackpix > 675.0
                    if pand <= 0.63:
                      return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if pand > 0.63
                      return [2, [0.0, 4.0, 0.0, 0.0, 0.0]]
  else:  # if height > 3.5
    if eccen <= 0.236:
      if eccen <= 0.186:
        if eccen <= 0.093:
          if eccen <= 0.028:
            if blackpix <= 37.5:
              return [2, [0.0, 1.0, 0.0, 1.0, 0.0]]
            else:  # if blackpix > 37.5
              return [4, [0.0, 0.0, 0.0, 15.0, 0.0]]
          else:  # if eccen > 0.028
            return [4, [0.0, 0.0, 0.0, 30.0, 0.0]]
        else:  # if eccen > 0.093
          if eccen <= 0.0975:
            return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
          else:  # if eccen > 0.0975
            if blackpix <= 100.5:
              if blackpix <= 7.5:
                if eccen <= 0.134:
                  return [4, [0.0, 0.0, 0.0, 1.0, 1.0]]
                else:  # if eccen > 0.134
                  return [4, [0.0, 1.0, 0.0, 5.0, 0.0]]
              else:  # if blackpix > 7.5
                if area <= 9.5:
                  if blackand <= 8.5:
                    return [4, [1.0, 0.0, 0.0, 6.0, 0.0]]
                  else:  # if blackand > 8.5
                    return [4, [1.0, 0.0, 0.0, 8.0, 0.0]]
                else:  # if area > 9.5
                  return [4, [0.0, 0.0, 0.0, 12.0, 0.0]]
            else:  # if blackpix > 100.5
              if length <= 17.5:
                return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
              else:  # if length > 17.5
                return [4, [0.0, 0.0, 0.0, 1.0, 0.0]]
      else:  # if eccen > 0.186
        if pand <= 0.6215:
          return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
        else:  # if pand > 0.6215
          if height <= 15.0:
            return [4, [0.0, 0.0, 0.0, 1.0, 0.0]]
          else:  # if height > 15.0
            return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
    else:  # if eccen > 0.236
      if height <= 27.5:
        if meantr <= 30.135:
          if blackpix <= 11.5:
            if pblack <= 0.241:
              if blackand <= 42.5:
                if meantr <= 9.5:
                  if pblack <= 0.16:
                    if pblack <= 0.098:
                      return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if pblack > 0.098
                      return [5, [0.0, 0.0, 0.0, 0.0, 11.0]]
                  else:  # if pblack > 0.16
                    if height <= 6.5:
                      if pand <= 0.6415:
                        if blackand <= 13.0:
                          return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                        else:  # if blackand > 13.0
                          return [5, [0.0, 0.0, 0.0, 0.0, 4.0]]
                      else:  # if pand > 0.6415
                        return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if height > 6.5
                      return [1, [3.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if meantr > 9.5
                  return [2, [0.0, 2.0, 0.0, 0.0, 0.0]]
              else:  # if blackand > 42.5
                return [1, [4.0, 0.0, 0.0, 0.0, 0.0]]
            else:  # if pblack > 0.241
              if area <= 11.0:
                if pblack <= 0.8875:
                  if pand <= 0.9375:
                    if pblack <= 0.75:
                      return [1, [2.0, 0.0, 0.0, 1.0, 0.0]]
                    else:  # if pblack > 0.75
                      if pand <= 0.8375:
                        return [1, [1.0, 0.0, 0.0, 1.0, 0.0]]
                      else:  # if pand > 0.8375
                        return [1, [1.0, 0.0, 0.0, 1.0, 0.0]]
                  else:  # if pand > 0.9375
                    return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if pblack > 0.8875
                  return [4, [0.0, 0.0, 0.0, 2.0, 0.0]]
              else:  # if area > 11.0
                if eccen <= 1.225:
                  if area <= 35.5:
                    if blackand <= 29.5:
                      if blackand <= 13.5:
                        if meantr <= 4.75:
                          if pand <= 0.7585:
                            if wbtrans <= 2.5:
                              return [5, [0.0, 0.0, 0.0, 0.0, 2.0]]
                            else:  # if wbtrans > 2.5
                              return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                          else:  # if pand > 0.7585
                            return [1, [3.0, 0.0, 0.0, 0.0, 0.0]]
                        else:  # if meantr > 4.75
                          if pblack <= 0.6215:
                            return [1, [16.0, 0.0, 0.0, 0.0, 0.0]]
                          else:  # if pblack > 0.6215
                            if pblack <= 0.655:
                              return [4, [0.0, 0.0, 0.0, 1.0, 0.0]]
                            else:  # if pblack > 0.655
                              return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
                      else:  # if blackand > 13.5
                        return [1, [37.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if blackand > 29.5
                      if blackpix <= 10.0:
                        return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
                      else:  # if blackpix > 10.0
                        return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                  else:  # if area > 35.5
                    if wbtrans <= 4.0:
                      if height <= 7.0:
                        return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                      else:  # if height > 7.0
                        return [5, [0.0, 0.0, 0.0, 0.0, 2.0]]
                    else:  # if wbtrans > 4.0
                      return [1, [5.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if eccen > 1.225
                  if area <= 22.0:
                    return [5, [0.0, 0.0, 0.0, 0.0, 2.0]]
                  else:  # if area > 22.0
                    if pand <= 0.9285:
                      if eccen <= 1.875:
                        return [1, [5.0, 0.0, 0.0, 0.0, 0.0]]
                      else:  # if eccen > 1.875
                        if blackand <= 21.0:
                          return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                        else:  # if blackand > 21.0
                          return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if pand > 0.9285
                      return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
          else:  # if blackpix > 11.5
            if pand <= 0.15:
              if area <= 2544.0:
                return [5, [0.0, 0.0, 0.0, 0.0, 3.0]]
              else:  # if area > 2544.0
                return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
            else:  # if pand > 0.15
              if pand <= 0.5225:
                if eccen <= 2.155:
                  if pblack <= 0.1565:
                    if pand <= 0.332:
                      if blackand <= 34.5:
                        return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                      else:  # if blackand > 34.5
                        return [1, [3.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if pand > 0.332
                      if eccen <= 2.0105:
                        return [5, [0.0, 0.0, 0.0, 0.0, 13.0]]
                      else:  # if eccen > 2.0105
                        return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                  else:  # if pblack > 0.1565
                    if height <= 23.0:
                      if pand <= 0.5165:
                        if meantr <= 1.9:
                          if pand <= 0.488:
                            return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
                          else:  # if pand > 0.488
                            return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                        else:  # if meantr > 1.9
                          return [1, [21.0, 0.0, 0.0, 0.0, 0.0]]
                      else:  # if pand > 0.5165
                        return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                    else:  # if height > 23.0
                      return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                else:  # if eccen > 2.155
                  if meantr <= 5.64:
                    if height <= 4.5:
                      return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                    else:  # if height > 4.5
                      if blackand <= 56.5:
                        if blackand <= 55.5:
                          if pblack <= 0.134:
                            if area <= 178.5:
                              return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                            else:  # if area > 178.5
                              return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
                          else:  # if pblack > 0.134
                            return [1, [7.0, 0.0, 0.0, 0.0, 0.0]]
                        else:  # if blackand > 55.5
                          return [4, [0.0, 0.0, 0.0, 1.0, 0.0]]
                      else:  # if blackand > 56.5
                        return [1, [165.0, 0.0, 0.0, 0.0, 0.0]]
                  else:  # if meantr > 5.64
                    if height <= 7.5:
                      if blackpix <= 54.5:
                        return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                      else:  # if blackpix > 54.5
                        return [2, [0.0, 3.0, 0.0, 0.0, 0.0]]
                    else:  # if height > 7.5
                      if blackpix <= 259.0:
                        if length <= 38.0:
                          return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                        else:  # if length > 38.0
                          return [5, [0.0, 0.0, 0.0, 0.0, 2.0]]
                      else:  # if blackpix > 259.0
                        return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
              else:  # if pand > 0.5225
                if length <= 544.5:
                  if eccen <= 0.3925:
                    if area <= 36.0:
                      return [1, [3.0, 0.0, 0.0, 0.0, 0.0]]
                    else:  # if area > 36.0
                      return [5, [0.0, 0.0, 0.0, 0.0, 2.0]]
                  else:  # if eccen > 0.3925
                    if pblack <= 0.104:
                      if pand <= 0.6855:
                        return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
                      else:  # if pand > 0.6855
                        return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                    else:  # if pblack > 0.104
                      if meantr <= 1.08:
                        if wbtrans <= 14.5:
                          return [1, [3.0, 0.0, 0.0, 0.0, 0.0]]
                        else:  # if wbtrans > 14.5
                          return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                      else:  # if meantr > 1.08
                        if blackand <= 22.5:
                          if pand <= 0.6335:
                            if area <= 34.0:
                              return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                            else:  # if area > 34.0
                              return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                          else:  # if pand > 0.6335
                            return [1, [13.0, 0.0, 0.0, 0.0, 0.0]]
                        else:  # if blackand > 22.5
                          if pblack <= 0.1375:
                            if meantr <= 2.565:
                              return [1, [33.0, 0.0, 0.0, 0.0, 0.0]]
                            else:  # if meantr > 2.565
                              return [5, [0.0, 0.0, 0.0, 0.0, 2.0]]
                          else:  # if pblack > 0.1375
                            if area <= 8709.0:
                              if height <= 24.5:
                                if wbtrans <= 4.5:
                                  if blackand <= 90.0:
                                    return [1, [48.0, 0.0, 0.0, 0.0, 0.0]]
                                  else:  # if blackand > 90.0
                                    return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                                else:  # if wbtrans > 4.5
                                  if pand <= 0.5835:
                                    if pand <= 0.5825:
                                      return [1, [197.0, 0.0, 0.0, 0.0, 0.0]]
                                    else:  # if pand > 0.5825
                                      if eccen <= 1.8:
                                        if pblack <= 0.211:
                                          return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                                        else:  # if pblack > 0.211
                                          return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
                                      else:  # if eccen > 1.8
                                        return [1, [5.0, 0.0, 0.0, 0.0, 0.0]]
                                  else:  # if pand > 0.5835
                                    if blackand <= 89.5:
                                      if area <= 127.0:
                                        return [1, [659.0, 0.0, 0.0, 0.0, 0.0]]
                                      else:  # if area > 127.0
                                        if pand <= 0.6875:
                                          return [1, [13.0, 0.0, 0.0, 0.0, 0.0]]
                                        else:  # if pand > 0.6875
                                          return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                                    else:  # if blackand > 89.5
                                      return [1, [3444.0, 0.0, 0.0, 0.0, 0.0]]
                              else:  # if height > 24.5
                                if pblack <= 0.2165:
                                  if area <= 4322.5:
                                    return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                                  else:  # if area > 4322.5
                                    return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                                else:  # if pblack > 0.2165
                                  return [1, [29.0, 0.0, 0.0, 0.0, 0.0]]
                            else:  # if area > 8709.0
                              if meantr <= 12.39:
                                return [1, [9.0, 0.0, 0.0, 0.0, 0.0]]
                              else:  # if meantr > 12.39
                                return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
                else:  # if length > 544.5
                  return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
        else:  # if meantr > 30.135
          if eccen <= 6.487:
            if blackand <= 33.5:
              return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
            else:  # if blackand > 33.5
              return [1, [14.0, 0.0, 0.0, 0.0, 0.0]]
          else:  # if eccen > 6.487
            if meantr <= 33.74:
              if eccen <= 22.0325:
                return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
              else:  # if eccen > 22.0325
                return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
            else:  # if meantr > 33.74
              return [2, [0.0, 17.0, 0.0, 0.0, 0.0]]
      else:  # if height > 27.5
        if pblack <= 0.3015:
          if eccen <= 3.812:
            if eccen <= 2.6985:
              if eccen <= 0.4845:
                if height <= 69.0:
                  return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]
                else:  # if height > 69.0
                  return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
              else:  # if eccen > 0.4845
                return [5, [0.0, 0.0, 0.0, 0.0, 39.0]]
            else:  # if eccen > 2.6985
              if meantr <= 3.575:
                return [1, [2.0, 0.0, 0.0, 0.0, 0.0]]
              else:  # if meantr > 3.575
                if height <= 29.5:
                  return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if height > 29.5
                  return [5, [0.0, 0.0, 0.0, 0.0, 9.0]]
          else:  # if eccen > 3.812
            if pblack <= 0.2055:
              if blackpix <= 660.0:
                return [2, [0.0, 1.0, 0.0, 0.0, 0.0]]
              else:  # if blackpix > 660.0
                if eccen <= 4.598:
                  return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if eccen > 4.598
                  return [5, [0.0, 0.0, 0.0, 0.0, 5.0]]
            else:  # if pblack > 0.2055
              if height <= 46.0:
                return [1, [7.0, 0.0, 0.0, 0.0, 0.0]]
              else:  # if height > 46.0
                if length <= 320.0:
                  return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if length > 320.0
                  return [5, [0.0, 0.0, 0.0, 0.0, 2.0]]
        else:  # if pblack > 0.3015
          if eccen <= 1.2935:
            if meantr <= 7.035:
              if pand <= 0.728:
                return [3, [0.0, 0.0, 4.0, 0.0, 0.0]]
              else:  # if pand > 0.728
                if length <= 75.0:
                  return [1, [3.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if length > 75.0
                  return [3, [0.0, 0.0, 2.0, 0.0, 0.0]]
            else:  # if meantr > 7.035
              return [3, [0.0, 0.0, 15.0, 0.0, 0.0]]
          else:  # if eccen > 1.2935
            if wbtrans <= 944.0:
              if eccen <= 1.8055:
                if blackand <= 3771.0:
                  return [1, [4.0, 0.0, 0.0, 0.0, 0.0]]
                else:  # if blackand > 3771.0
                  if height <= 85.0:
                    return [3, [0.0, 0.0, 2.0, 0.0, 0.0]]
                  else:  # if height > 85.0
                    return [1, [1.0, 0.0, 0.0, 0.0, 0.0]]
              else:  # if eccen > 1.8055
                return [1, [14.0, 0.0, 0.0, 0.0, 0.0]]
            else:  # if wbtrans > 944.0
              if pand <= 0.6055:
                return [3, [0.0, 0.0, 5.0, 0.0, 0.0]]
              else:  # if pand > 0.6055
                return [5, [0.0, 0.0, 0.0, 0.0, 1.0]]

def prove_prediction():
    successful_attempts=0
    unsuccessful_attempts =0
    for index, row in dataset.iterrows():
        target = row[target_variable]
        features = row[feature_cols]
        prediction = predict(features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7], features[8], features[9])
        if(prediction[0] == int(target[0])):
            successful_attempts = successful_attempts + 1
        else:
            unsuccessful_attempts = unsuccessful_attempts + 1
    total = successful_attempts + unsuccessful_attempts
    accuracy = successful_attempts/total *100
    return 'El porcentaje de exito de las reglas es de: '+str(accuracy)+'% con un total de aciertos: '+str(successful_attempts)+'/'+str(total)+' y un total de errores: '+str(unsuccessful_attempts)+'/'+str(total)

result = prove_prediction()

#Abrimos el archivo para extenderlo
file = open("DaRules.txt", "a")
file.write('\nEXACTITUD: \n')
file.write(result)
file.close()

#abrimos el archivo en el sistema operativo
file_dir = "DaRules.txt"
# Verifica si el archivo existe
if os.path.exists(file_dir):
    # Abre el archivo en el programa asociado
    if os.name == "nt":
        os.system("start " + file_dir)
    elif os.name == "posix":
        os.system("xdg-open " + file_dir)
else:
    print("El archivo no existe.")


