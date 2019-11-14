# -*- coding: utf-8 -*-

import time
import Tictactoe 
from random import randint,choice

def RandomMove(b):
    '''Renvoie un mouvement au hasard sur la liste des mouvements possibles'''
    return choice(b.legal_moves())

def deroulementRandom(b):
    '''Effectue un déroulement aléatoire du jeu de morpion.'''
    print("----------")
    print(b)
    if b.is_game_over():
        res = getresult(b)
        if res == 1:
            print("Victoire de X")
        elif res == -1:
            print("Victoire de O")
        else:
            print("Egalité")
        return
    b.push(RandomMove(b))
    deroulementRandom(b)
    b.pop()


def getresult(b):
    '''Fonction qui évalue la victoire (ou non) en tant que X. Renvoie 1 pour victoire, 0 pour 
       égalité et -1 pour défaite. '''
    if b.result() == b._X:
        return 1
    elif b.result() == b._O:
        return -1
    else:
        return 0

# def minmax(b, player):
#     if b.is_game_over():
#         return getresult(b)
#     if player == 1:  #1 est le joueur ami
#         value = -2 #valeur a maximiser
#         for i in b.legal_moves():
#             b.push(i)
#             value = max(value, minmax(b, player*-1))
#             b.pop()
#         return value
#     if player == -1:  #-1 est le joueur ennemi
#         value = 2   #valeur a minimiser
#         for i in b.legal_moves():
#             b.push(i)
#             value = min(value, minmax(b, player*-1))
#             b.pop()
#         return value
#     return

def minmax(b, player):
    if b.is_game_over():
        return getresult(b)
    if player == 1:  #1 est le joueur ami
        value = -2 #valeur a maximiser
        for i in b.legal_moves():
            b.push(i)
            oldvalue=value
            value = max(value, minmax(b, player*-1))
            #if(oldvalue!=value)
             #   self._trace[] = i
            b.pop()
        return value
    if player == -1:  #-1 est le joueur ennemi
        value = 2   #valeur a minimiser
        for i in b.legal_moves():
            b.push(i)
            value = min(value, minmax(b, player*-1))
            b.pop()
        return value
    return



def cpt(b):
    print(b)
    if b.is_game_over():
        return 1
    count=0
    for z in b.legal_moves():
        b.push(z)
        count+=cpt(b)
        b.pop()
    return count

board = Tictactoe.Board()
print(board)

### Deroulement d'une partie aléatoire

print(minmax(board, 1))



print("Apres le match, chaque coup est défait (grâce aux pop()): on retrouve le plateau de départ :")
print(board)

