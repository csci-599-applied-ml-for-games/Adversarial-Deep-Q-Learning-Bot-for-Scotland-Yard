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




# AutoListCtrl.py
#
# This module contains a base class list control that does the following:
#     * sort by column when clicking on headers (ColumnSorterMixin)
#     * auto-expands the width of the last column to fill available space
#       (ListCtrlAutoWidthMixin)
#     * supports realtime addition and removal of items
#
# This base class will be used in both the game room browser and the
# team selection window.
#


# appears after first logging in to a server.  Users may join exactly one game
# at a time, at which point a player registration window is spawned.


import os.path, gettext, wx
from twisted.python import log
import wx
from wx.lib.mixins.listctrl import ColumnSorterMixin, ListCtrlAutoWidthMixin
from londonlaw.common.config import *



# the AutoWidthMixin simply resizes the last column of of the
# ListCtrl to take up all remaining space.
class AutoWidthListCtrl(wx.ListCtrl, ListCtrlAutoWidthMixin):
   def __init__(self, parent, ID, pos = wx.DefaultPosition,
         size = wx.DefaultSize, style = 0):
      wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
      ListCtrlAutoWidthMixin.__init__(self)


# 'headers' is a list of column headers.
# 'placeholder' should be a list of display data that is shown when
# the list is empty (same length as 'headers').
class AutoListCtrl(AutoWidthListCtrl, ColumnSorterMixin):
   def __init__(self, parent, ID, headers, placeholder = None):
      AutoWidthListCtrl.__init__(self, parent, ID, wx.DefaultPosition, wx.DefaultSize,
            wx.LC_REPORT|wx.LC_SINGLE_SEL)

      self.headers = headers

      # load in the tiny arrow images that ColumnSorterMixin draws
      # in the headers of sorted columns
      # WARNING: this segfaults if imageList is a local variable.
      # Maybe a wxPython bug... imageList falls out of scope and gets deleted prematurely?
      self.imageList = wx.ImageList(16, 16, True)
      file1 = os.path.normpath(os.path.join(MEDIAROOT, "images/smalluparrow.png"))
      file2 = os.path.normpath(os.path.join(MEDIAROOT, "images/smalldownarrow.png"))
      image = wx.Image(file1, wx.BITMAP_TYPE_ANY)
      image.SetMaskColour(255, 255, 255)
      self.smallUpArrow = self.imageList.Add(wx.BitmapFromImage(image))
      image = wx.Image(file2, wx.BITMAP_TYPE_ANY)
      image.SetMaskColour(255, 255, 255)
      self.smallDnArrow = self.imageList.Add(wx.BitmapFromImage(image))
      self.SetImageList(self.imageList, wx.IMAGE_LIST_SMALL)

      self.placeholder = placeholder
      # data from the server should be formatted as
      # ("game name", "game type", "game status", "number of players")
      if self.placeholder:
         self.itemDataMap = {0 : self.placeholder}
      else:
         self.itemDataMap = {}
      self.populateList() 

      # this must be called *after* the list has been created
      ColumnSorterMixin.__init__(self, len(self.headers)) 


   def populateList(self):
      info          = wx.ListItem()
      info.m_mask   = wx.LIST_MASK_TEXT | wx.LIST_MASK_IMAGE | wx.LIST_MASK_FORMAT
      info.m_image  = -1
      info.m_format = wx.LIST_FORMAT_CENTRE

      for i in range(len(self.headers)):
         info.m_text = self.headers[i]
         self.InsertColumnInfo(i, info)

      items = self.itemDataMap.items()
      for i in range(len(items)):
         key, data = items[i]
         self.InsertStringItem(i, data[0])
         for j in range(1, len(self.headers)):
            self.SetStringItem(i, j, data[j])
         self.SetItemData(i, key)

      # dirty hack... wxWidgets needs a wx.LIST_AUTOSIZE_* that
      # chooses the maximum of BOTH header size and list item size
      for i in range(len(self.headers) - 1):
         self.SetColumnWidth(i, wx.LIST_AUTOSIZE) 
         itemWidth = self.GetColumnWidth(i)
         self.SetColumnWidth(i, wx.LIST_AUTOSIZE_USEHEADER)
         headerWidth = self.GetColumnWidth(i)
         if headerWidth < itemWidth:
            self.SetColumnWidth(i, wx.LIST_AUTOSIZE) 
      # size of last column is set automatically by ListCtrlAutoWidthMixin


#   def logListData(self):
#      print "itemDataMap = " + str(self.itemDataMap)
#      for i in range(len(self.itemDataMap)):
#         print "item[" + `i` + "]      = " + `self.GetItemData(i)` + ", " + \
#            self.GetItemText(i)


   # this will either add a new item or update an existing item if the first
   # piece of data matches
   def addItem(self, data):
      log.msg("called GameListWindow.addGameInfo(), data = " + str(data))
      # check for pre-existing matching data
      foundMatch = False
      for item in range(len(self.itemDataMap)):
         if data[0] == self.itemDataMap[item][0]:
            foundMatch = True
            for i in range(1, len(data)):
               self.SetStringItem(item, i, data[i])

      if not foundMatch:
         # if no matches found, add a new item to the list
         if self.placeholder:
            if self.itemDataMap[0] == self.placeholder:
               log.msg("deleting placeholder")
               self.DeleteItem(0)
               del self.itemDataMap[0]
         index = len(self.itemDataMap)
         self.itemDataMap[index] = data
         self.InsertStringItem(index, data[0])
         for i in range(1, len(self.headers)):
            self.SetStringItem(index, i, data[i])
         self.SetItemData(index, index)


   # Note: This is completely asinine.  Removing an item should be an easy
   #       process that does not impact sorting.
   # This routine will remove the item with matching data[0] value.
   def removeItemByData(self, data):
      for key in self.itemDataMap:
         if self.itemDataMap[key][0] == data[0]:
            # update the datamap 
            for i in range(key, len(self.itemDataMap)-1):
               self.itemDataMap[i] = self.itemDataMap[i+1]
            del self.itemDataMap[len(self.itemDataMap)-1]
            if len(self.itemDataMap) == 0 and self.placeholder:
               self.itemDataMap[0] = self.placeholder
               self.InsertStringItem(0, self.placeholder[0])
               for i in range(len(self.headers)):
                  self.SetStringItem(0, i, self.placeholder[i])
            break

      # delete the requested item
      for item in range(self.GetItemCount()):
         if self.GetItemText(item) == data[0]:
            self.DeleteItem(item)
            break
      # update the ItemData field associated with the remaining items,
      # to keep them in sync with the keys of the itemDataMap
      for item in range(len(self.itemDataMap)):
         for key in range(len(self.itemDataMap)):
            if self.itemDataMap[key][0] == self.GetItemText(item):
               self.SetItemData(item, key)


   # required by ColumnSorterMixin
   def GetListCtrl(self):
      return self


   # used by ColumnSorterMixin to display up and down arrows
   # on sorted column headers
   def GetSortImages(self):
      return (self.smallDnArrow, self.smallUpArrow)



