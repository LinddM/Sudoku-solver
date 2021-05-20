import numpy as np

# normalizar la data
def norm(a):
    return (a/9)-.5

# desnormalizar
def denorm(a):
    return (a+.5)*9

def inference_sudoku(sample, model):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    feat = sample.copy() # sudoku de 9x9

    while(1): # repite el proceso hasta que ya no hayan celdas vacias
    
        out = model.predict(feat.reshape((1,9,9,1))) # prediccion del modelo. Me trae un arreglo de la probabilidad que sea x numero en todos los espacios del sudoku
        out = out.squeeze() # (81, 9) quitarle dimension 1

        pred = np.argmax(out, axis=1).reshape((9,9))+1 # returns the indices of the maximum values along an axis
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) # ve las probabilidades maximas y las redondea en una matriz
        
        feat = denorm(feat).reshape((9,9)) # desnormalizar el sudoku
        mask = (feat==0) # ver espacios vacios
        
        if(mask.sum()==0):
            break # indica que ya todas las celdas estan vacias
            
        prob_new = prob*mask # deja 0 en los que ya estan llenos

        ind = np.argmax(prob_new) # toma la primera probabilidad mas alta
        x, y = (ind//9), (ind%9) # toma el indice de la celda; floor division, modulus

        val = pred[x][y] # el numero predicho en ese indice
        feat[x][y] = val
        feat = norm(feat) # lo normaliza otra vez
    
    return pred

def test_accuracy(feats, labels, model):
    
    correct = 0
    
    for i, feat in enumerate(feats):
        pred = inference_sudoku(feat, model) # test sudokus (para llenar)
        
        true = labels[i].reshape((9,9))+1 # soluciones reales
        
        if(abs(true - pred).sum()==0): # corroborar si la solucion es correcta
            correct += 1
        
    print(correct/feats.shape[0]) # proporcion de correctas vs numero total de sudokus

# https://github.com/shivaverma/Sudoku-Solver