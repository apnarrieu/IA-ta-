import time
import chess
from random import randint, choice

pieces_values = {'P':1,'R':2,'N':2,'B':2,'Q':4,'K':10,'p':-1,'r':-2,'n':-2,'b':-2,'q':-4,'k':-10}

def randomMove(b):
    '''Renvoie un mouvement au hasard sur la liste des mouvements possibles. Pour avoir un choix au hasard, il faut
    construire explicitement tous les mouvements. Or, generate_legal_moves() nous donne un itérateur.'''
    return choice([m for m in b.generate_legal_moves()])

def deroulementRandom(b):
    '''Déroulement d'une partie d'échecs au hasard des coups possibles. Cela va donner presque exclusivement
    des parties très longues et sans gagnant. Cela illustre cependant comment on peut jouer avec la librairie
    très simplement.'''
    print("----------")
    print(b)
    if b.is_game_over():
        print("Resultat : ", b.result())
        return
    b.push(randomMove(b))
    deroulementRandom(b)
    b.pop()

def heuristic(b):
    x=0;
    for i in b.piece_map():
        x+=pieces_values[b.piece_map()[i].symbol()]
    return x
    
    
def minmax(b, player, depth):
    if depth==0:
        return heuristic(b)
    if player == 1:  #1 est le joueur ami
        value = -10000 #valeur a maximiser
        for i in b.generate_legal_moves():
            b.push(i)
            value = max(value, minmax(b, player*-1,depth-1))
            b.pop()
            return value
    if player == -1:  #-1 est le joueur ennemi
        value = 10000   #valeur a minimiser
        for i in b.generate_legal_moves():
            b.push(i)
            value = min(value, minmax(b, player*-1,depth-1))
            b.pop()
            return value
    return

board = chess.Board()
i = minmax(board, 1, 6)
print(i)

#deroulementRandom(board)

