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



import random, sets
from twisted.internet import protocol
from twisted.python import log
from londonlaw.aiclients import base
from londonlaw.common import path
import detectives_dqn


class DetectiveSimpleAIProtocolError(Exception):
   pass


  


# A very simple but complete AI client for the detectives.
class DetectiveSimpleAIProtocol(base.BaseAIProtocol):

   def generate_feature_space(self, pawnName):
      feature_vector = []
      # log.msg("generate feature space for DETECTIVES called")
      log.msg(self._history)
      mrXLoc = self._pawns['X'].getLocation()
      if(mrXLoc == -1):
         mrXLoc = 0
      feature_vector.append(mrXLoc)
      # log.msg("Turn number: ", self._turnNum)
      # log.msg("Last known location of Mr. X: ", mrXLoc)
      if(self._turnNum <= 3):
         number_of_turns_still_x_reveals_location = 3 - self._turnNum
      elif(self._turnNum <= 8):
         number_of_turns_still_x_reveals_location = 8 - self._turnNum
      elif(self._turnNum <= 13):
         number_of_turns_still_x_reveals_location = 13 - self._turnNum
      elif(self._turnNum <= 18):
         number_of_turns_still_x_reveals_location = 18 - self._turnNum
      else:
         number_of_turns_still_x_reveals_location = 24 - self._turnNum
      feature_vector.append(number_of_turns_still_x_reveals_location)
      # log.msg(" number_of_turns_still_x_reveals_location: ", number_of_turns_still_x_reveals_location)
      dets      = ['Red', 'Yellow', 'Green', 'Blue', 'Black']
      detLocs   = [self._pawns[d].getLocation() for d in dets]
      # log.msg("Location of all detectives: ", detLocs)
      feature_vector.append(detLocs)
      det_tickets = [self._pawns[d]._tickets for d in dets]
      det_tickets_vector = [list(d.values())[:3] for d in det_tickets]
      feature_vector.append(det_tickets_vector)
      # log.msg("Resources used by the detectives: ", det_tickets_vector)
      xTransports = []
      if(self._turnNum <= 5):
         for i in range(0, self._turnNum):
            if(self._history[i+1]['X'][1] == 'taxi'):
               xTransports.append(1)
            elif(self._history[i+1]['X'][1] == 'bus'):
               xTransports.append(2)
            else:
               xTransports.append(3)
         for i in range(self._turnNum, 5):
            xTransports.append(0)
      else:
         for i in range(self._turnNum - 5, self._turnNum):
            if(self._history[i+1]['X'][1] == 'taxi'):
               xTransports.append(1)
            elif(self._history[i+1]['X'][1] == 'bus'):
               xTransports.append(2)
            else:
               xTransports.append(3)
      # log.msg("Resources used by Mr. X in the last 5 turns: ", xTransports)
      feature_vector.append(xTransports)
      # log.msg("Detective number: ", self._pawns[pawnName]._name)
      if(self._pawns[pawnName]._name == "Red"):
         feature_vector.append(1)
      if(self._pawns[pawnName]._name == "Yellow"):
         feature_vector.append(2)
      if(self._pawns[pawnName]._name == "Green"):
         feature_vector.append(3)
      if(self._pawns[pawnName]._name == "Blue"):
         feature_vector.append(4)
      if(self._pawns[pawnName]._name == "Black"):
         feature_vector.append(5)
      log.msg("Input feature vector: ", feature_vector)
      return feature_vector

   def doTurn(self, pawnName):
      state = self.generate_feature_space(pawnName)
      bestMove, bestTransport = detectives_dqn.predict_best_move(state)
      self.makeMove([pawnName.lower(), bestMove, bestTransport)
      log.msg("Best Move for ", pawnName.lower(), "detective is to", bestMove, "using", bestTransport)


   def response_ok_tryjoin(self, tag, args):
      self._state = "trychat"
      self.sendChat("Loaded \"Simple Mr. X AI\"", "all")


   def response_ok_trychat(self, tag, args):
      base.BaseAIProtocol.response_ok_tryjoin(self, tag, args)

      


class DetectiveSimpleAIFactory(base.BaseAIFactory):
   protocol = DetectiveSimpleAIProtocol

   def __init__(self, username, gameroom):
      base.BaseAIFactory.__init__(self, username, username, gameroom, "Detectives")




