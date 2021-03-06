#  London Law -- a networked manhunting board game
#  Copyright (C) 2003-2004, 2005 Paul Pelzl
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 2, as 
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



import random
from twisted.internet import protocol
from twisted.python import log
from londonlaw.aiclients import base
from londonlaw.common import path
import x_dqn

# A very simple but complete AI client for Mr. X.
class XSimpleAIProtocol(base.BaseAIProtocol):

   # Look in the list of safe moves.  Find a move which is as safe as
   # possible to move to, but make black tickets expensive to use.
   def one_hot_encode(self, value, size, max_value):
      import numpy as np
      a = np.array(value)
      b = np.zeros((size, max_value))
      b[np.arange(size), a] = 1
      return b.tolist()

  

   def generate_feature_space(self):
      log.msg("generate feature space called")
      mrXLoc = self._pawns['X'].getLocation()
      #mrXLoc_vector = self.one_hot_encode([mrXLoc], 1, 199)
      mrXTickets = self._pawns['X']._tickets
      mrXTickets_vector = list(mrXTickets.values())[-2]
      dets      = ['Red', 'Yellow', 'Green', 'Blue', 'Black']
      detLocs   = [self._pawns[d].getLocation() for d in dets]
      #detLocs_vector = self.one_hot_encode(detLocs, 5, 199)
      det_tickets = [self._pawns[d]._tickets for d in dets]
      det_tickets_vector = [list(d.values())[:3] for d in det_tickets]
      turn_number = self._turnNum
      feature_vector = [mrXLoc, mrXTickets_vector, detLocs, det_tickets_vector, turn_number]
      log.msg("Input feature vector: ", feature_vector)
      return feature_vector
  

   def doTurn(self, pawnName):

      log.msg("\n\n\n") 
      log.msg("Mr. X location: ", self._pawns['X'].getLocation()) 

      log.msg("Mr. X Tickets: "+ str(self._pawns['X']._tickets)+"\n")
      dets      = ['Red', 'Yellow', 'Green', 'Blue', 'Black']
      detLocs   = [self._pawns[d].getLocation() for d in dets]
      log.msg("DETETIVE LOCATIONS: "+ str(detLocs))
      det_tickets = [self._pawns[d]._tickets for d in dets]
      log.msg("DETECTIVE TICKETS: "+ str(det_tickets)+"\n")
      log.msg("Turn number: "+ str(self._turnNum)+"\n")
      # self.turnNo += 1
      # log.msg("Turn No: "+ str(self.turnNo)+"\n")
      state = self.generate_feature_space()
      bestMove, bestTransport = x_dqn.predict_best_move(state)

      log.msg("Best move is from: ", self._pawns['X'].getLocation(), "to : ", bestMove, "using: ",  bestTransport)
      self.makeMove(['x', str(bestMove), str(bestTransport)])
      log.msg("making a move " +"\n\n\n") 

   def response_ok_tryjoin(self, tag, args):
      self._state = "trychat"
      self.sendChat("Loaded \"Simple Mr. X AI\"", "all")
   
  

   

   def response_ok_trychat(self, tag, args):
      base.BaseAIProtocol.response_ok_tryjoin(self, tag, args)

      


class XSimpleAIFactory(base.BaseAIFactory):
   protocol = XSimpleAIProtocol

   def __init__(self, username, gameroom):
      base.BaseAIFactory.__init__(self, username, username, gameroom, "Mr. X")
      
      move_file = "/Users/shreyasi/Desktop/LondonLaw/move.txt"



