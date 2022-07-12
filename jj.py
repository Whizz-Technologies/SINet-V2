# Generated with SMOP  0.41-beta
from libsmop import *
# 

    ##
    
@function
def Fmeasure_calu(sMap=None,gtMap=None,gtsize=None,threshold=None,*args,**kwargs):
    varargin = Fmeasure_calu.varargin
    nargin = Fmeasure_calu.nargin

    #threshold =  2* mean(sMap(:)) ;
    if (threshold > 1):
        threshold=1
# .\eval\Fmeasure_calu.m:5
    
    Label3=zeros(gtsize)
# .\eval\Fmeasure_calu.m:8
    Label3[sMap >= threshold]=1
# .\eval\Fmeasure_calu.m:9
    NumRec=length(find(Label3 == 1))
# .\eval\Fmeasure_calu.m:11
    LabelAnd=logical_and(Label3,gtMap)
# .\eval\Fmeasure_calu.m:12
    NumAnd=length(find(LabelAnd == 1))
# .\eval\Fmeasure_calu.m:13
    num_obj=sum(sum(gtMap))
# .\eval\Fmeasure_calu.m:14
    if NumAnd == 0:
        PreFtem=0
# .\eval\Fmeasure_calu.m:17
        RecallFtem=0
# .\eval\Fmeasure_calu.m:18
        FmeasureF=0
# .\eval\Fmeasure_calu.m:19
    else:
        PreFtem=NumAnd / NumRec
# .\eval\Fmeasure_calu.m:21
        RecallFtem=NumAnd / num_obj
# .\eval\Fmeasure_calu.m:22
        FmeasureF=((dot(dot(1.3,PreFtem),RecallFtem)) / (dot(0.3,PreFtem) + RecallFtem))
# .\eval\Fmeasure_calu.m:23
    
    #Fmeasure = [PreFtem, RecallFtem, FmeasureF];
    