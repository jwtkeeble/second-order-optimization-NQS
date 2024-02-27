import pandas as pd

from typing import Optional

class WriteToFile(object):
  """
  Class to write a dict to a Pandas' dataframe and saved to a .csv
  """

  def __init__(self, load: Optional[str], filename: str) -> None:
    r"""Writer Object
    """
    self.dataframe=None
    if(isinstance(load, str)):
      self.load(load)
    self._data = [] #append data to list, and push to dataframe when writing!

  def load(self, filename: str) -> None:
    r"""Method to load an existing .csv file in which to write.

        :param filename: The filename to which the data is saved
        :type filename: str
    """
    self.dataframe = pd.read_csv(filename, index_col=[0]) #index_col needed anymore? (check with Adam opt)

  def write_to_file(self, filename: str) -> None:
    r"""Method to write current dataframe to file with given filename

        :param filename: The filename to which the data is saved
        :type filename: str
    """
    #if(isinstance(self.dataframe, pd.DataFrame)):
    newDataFrame = pd.DataFrame.from_records(self._data, index='epoch') #creates data frame for new '_data'
    self.dataframe = pd.concat([self.dataframe, newDataFrame]) #concatenates to existing dataframe.

    self.dataframe.to_csv(filename) #writes to csv
    self._data=[] #reset data (to avoid re-appending data)

  def __call__(self, dic: dict) -> None:
    r"""Method to write to file by concatenating a new `pd.DataFrame` object
        to the existing `pd.DataFrame` object. The current `pd.DataFrame` object
        is stored as a class attribute and continually updated via the `__call__` method.

        :param dic: A Dict object containing the properties being saved (along with their corresponding values)
        :type dic: dict
    """
    self._data.append(dic)

    """
    if(self.dataframe is None):
      self.dataframe = pd.DataFrame.from_dict(dic)
      self.dataframe.set_index('epoch',inplace=True) #set index to epochs
    else:
      row = pd.DataFrame.from_dict(dic)
      row.set_index('epoch',inplace=True) #set index to epoch
      self.dataframe = pd.concat([self.dataframe, row]) #TODO: change, this is slow!
    """