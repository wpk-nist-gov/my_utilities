#
#placeholder for general utility functions
#

def default_dic(defaults,actual,inplace=False,clobber=False):
    """
    return a composit of defaults and actual dictionaries
    """


    #make a copy of actual
    if inplace:
        if clobber:
            defaults.clear()
        defaults.update(actual)
    else:
        if clobber:
            return actual.copy()
        else:
            return dict(defaults,**actual)

    # #loop through defaults
    # #if value in final, use that, otherwise use default
    # for k,v in defaults.iteritems():
    #     final[k]=final.get(k,v)

    # return final



# class DefDic(object):
#     """
#     default dictionary
#     """

#     def __init__(self,
#                  Clobber=False,
#                  **kwargs
#                  ):
#         """
#         define a default dictionary

#         *Parameters*

#         Dic: dict
#           default dictionary

#         Clobber: bool
#           whether to clobber defaults
#         """
#         self.Dic={}
#         self.set_Dic(Clobber,**kwargs)

#     def get_Dic(self,Clobber=False,**kwargs):
#         """
#         get combined Dictionary
#         """
#         if Clobber:
#             return kwargs
#         else:
#             return default_dic(self.Dic,kwargs)

#     def set_Dic(self,Clobber=False,**kwargs):
#         """
#         set default dictionary
#         """
#         self.Dic=self.get_Dic(Clobber,**kwargs)

#     def __repr__(self):
#         return repr(self.Dic)
