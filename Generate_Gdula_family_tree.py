#!/usr/bin/env python
# coding: utf-8

# ># Packages needed to import 

# In[123]:


import networkx as nx
import matplotlib.pyplot as plt
#import igraph
#from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import json
import urllib
import urllib.request
import html2text
import math
import pandas as pd
pd.options.mode.chained_assignment = None 
import re
import numpy as np
import networkx as nx
from matplotlib import pylab as pl
import time
from datetime import date

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout, write_dot
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout, write_dot
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")


# In[124]:


from ipywidgets import interactive
import pandas as pd


# # Gdula Family Genealogy reading and parsing

# In[125]:


start_time = time.time()

## Gets the most recent updates from the site
from bs4 import BeautifulSoup

def curr_update():
    x1 = urllib.request.urlopen('http://www.gdula.info/272-2')
    text1 = x1.read().decode(x1.info().get_content_charset('utf8'))

    ta = BeautifulSoup(text1, 'lxml').find_all('meta')

    for i in ta:
        if 'property' in i.attrs.keys():
            if i.attrs['property'] == 'article:modified_time':
                new_date = i.attrs['content']

    nday = int(new_date[8:10])
    nmonth = int(new_date[5:7])
    nyear = int(new_date[2:4])

    return str(nmonth)+'-'+str(nday)+'-'+str(nyear)

curr_time = curr_update()

# In[127]:


def mult_vals(col):
    orig = pd_gdula[col].tolist()
    l1,l2 = [],[]
    for i in range(len(orig)):
        v = orig[i]
        if isinstance(v, str):
            if ',  ' in orig[i]:
                c = orig[i].replace(',  ',';')
                orig[i] = c
            elif '  ' in orig[i]:
                c = orig[i].replace('  ',';')
                orig[i] = c
        elif math.isnan(v):
            orig[i] = ""

    for i in range(len(orig)):
        if ';' in orig[i]:
            k = orig[i].split(';')
            l1.append(k[0])
            l2.append(k[1])
        elif ';' not in orig[i] and orig[i] != "":
            l1.append(orig[i])
            l2.append("")
        else:
            l1.append('')
            l2.append('')
                
    return orig, l1, l2


# In[128]:


#curr_html = 'http://www.gdula.info/genealogy-'+curr_time+'/individuals.php'
pda = pd.read_html('http://www.gdula.info/genealogy-'+curr_time+'/individuals.php')
pd_gdula = pda[0]

#print(pd_gdula.head())


# In[129]:


def sing_val(col):
    give = pd_gdula[col].tolist()
    c=""
    for i in range(len(give)):
        m = give[i]
        if col != "Surname":
            v = give[i]
            if not isinstance(v, str):
                give[i] = ""
        else:
            if i == 0:
                give[i]=""
            elif isinstance(give[i], str):
                c = m
            elif math.isnan(give[i]):
                give[i] = c
    return give


# In[130]:


p_1,f_1,m_1 = mult_vals('Parents')
pd_gdula['Parents'],pd_gdula['Father'],pd_gdula['Mother']= p_1,f_1,m_1


# In[131]:


part_a,part_0,part_1 = mult_vals('Partner')
pd_gdula['Partner0'],pd_gdula['Partner1'] = part_0,part_1


# In[132]:


pd_gdula['Surname']=sing_val('Surname')


# In[133]:


pd_gdula['Given Name']=sing_val('Given Name')


# In[134]:


pd_gdula['Birth']=sing_val('Birth')


# In[135]:


pd_gdula['Death']=sing_val('Death')


# In[136]:


pd_gdula['Name'] = pd_gdula[['Surname', 'Given Name']].agg(', '.join, axis=1)
pd_gdula.drop(['Surname', 'Given Name','Partner','Parents'], axis = 1) 


# In[137]:


names = []
allnames = []
for i in range(len(pd_gdula)):
    p0 = pd_gdula.iloc[i,4]
    n = pd_gdula.iloc[i,6]
    allnames.append(n)
    if n not in names:
        names.append(n)
    else:
        dup = allnames.index(n)
        if p0 != pd_gdula.iloc[allnames.index(n),4]:
            r = pd_gdula.iloc[dup,0:7].tolist()
            if pd_gdula.iloc[i,4] == pd_gdula.iloc[dup,4]:
                print(pd_gdula.iloc[i,0:7].tolist())

# ok, so no people with the same name have partner0 with the same name  


# In[138]:


url = "http://www.gdula.info/genealogy-"+curr_time+"/individuals.php"
x = urllib.request.urlopen(url)

text = x.read().decode(x.info().get_content_charset('utf8'))
k = text.split('/thead')

s = [j.replace('<td>','').split('</td>') for j in k[1].split('tbody')]
s1 = [item for sublist in s for item in sublist]

#e_str,s_str = 'php','"'

#s2 = [s for s in s1 if 'php' in s]
approved = ['ColumnName','ColumnPartner']
s2 = [url for url in s1 if any(sub in url for sub in approved)]

c_n,c_p0,c_p1 = [],[],[]

for j in s2:
    if 'ColumnName' in j:
        if 'php' not in j:
            c_n.append('none')
        st,st1 = j.index('href="')+6,j.index('.php')+4
        c_n.append(j[st:st1])
    if 'ColumnPartner' in j:
        if 'php' not in j:
            c_p0.append('')
            c_p1.append('')
        elif j.count('.php') == 2:
            st,st1 = j.index('href="')+6,j.index('.php')+4
            c_p0.append(j[st:st1])
            sta,sta1 = j.index('href="',st1+1)+6,j.index('.php',st1+1)+4
            c_p1.append(j[sta:sta1])
        else:
            st,st1 = j.index('href="')+6,j.index('.php')+4
            c_p0.append(j[st:st1])
            c_p1.append('')

pd_gdula['NamePhp'],pd_gdula['Partner0Php'],pd_gdula['Partner1Php'] = c_n,c_p0,c_p1
#print(pd_gdula.head())


# In[171]:


def relative_app(arr):
    out = []
    for a in arr:
        k = list(set(a))
        #o = []
        if not k:
            out.append("")
        else:
            if len(k) == 1:
                out.append(k[0])
            else:
                out.append(k)
    #print(out)
    return out


# In[172]:


def get_corr_col(line,vals):
    curr_dict = {'">Father':0,'">Stepfather':2,'">Mother':1,'">Stepmother':3,';Brother<':4,';Half-brother<':4,';Sister<':4,';Half-sister<':4}
    for l in vals:
        r = l in line
        if r == True:
            #print(l)
            return curr_dict[l]


# In[173]:


c_f,c_sf,c_m,c_sm,c_sib = [],[],[],[],[]
o_string = "http://www.gdula.info/genealogy-"+curr_time+"/"
approved = ['">Father','">Stepfather','">Mother','">Stepmother',';Brother<',';Half-brother<',';Sister<',';Half-sister<']

for i in c_n:
    #print('new')
    x1 = urllib.request.urlopen(o_string+ i)
    text1 = x1.read().decode(x1.info().get_content_charset('utf8'))
    
    t = text1.split('<div class="content" id="IndividualDetail">')[1].split('<div class="fullclear"></div>')[0].split('</div>')

    m = [re.split('table class="infolist" |tr>\r\n',i) for i in t]
    m1 = [item for sublist in m for item in sublist]
    m2 = [s for s in m1 if len(s) >= len(o_string+i)]
    m3 = [url for url in m2 if any(sub in url for sub in approved)]

    all_vals = [[],[],[],[],[]]
    
    for k in m3:
        if 'Non Relative' not in k:
            ik = get_corr_col(k,approved)
            s1 = k.index('href="')+6
            e1 = k.index('.php')+4
            all_vals[ik].append(k[s1:e1].replace('../',''))
            
    a = relative_app(all_vals)
    
    c_f.append(a[0])
    c_m.append(a[1])
    c_sf.append(a[2])
    c_sm.append(a[3])
    c_sib.append(a[4])
    
pd_gdula['FatherPhp'],pd_gdula['StepFatherPhp'],pd_gdula['MotherPhp'],pd_gdula['StepMotherPhp'],pd_gdula['SiblingPhp'] = c_f,c_sf,c_m,c_sm,c_sib


# In[174]:


pd_gdula1 = pd_gdula.replace(regex=r'.php', value='')
#print(pd_gdula1.columns.tolist())
pd_gdula1 = pd_gdula1[['Birth','Death','Name','NamePhp','Partner0Php','Partner1Php','FatherPhp','StepFatherPhp','MotherPhp','StepMotherPhp','SiblingPhp']]
#print(pd_gdula1['Name'].tolist())


# ## Merge partners into one node

# In[175]:


def tuple_people(pr):
    tot = []
    for j in pr:
        tot.append(j[0])
        n = len(j)
        for k in range(1,n):
            if j[k] != "":
                tot.append(j[k])
        
    return list(set(tot))

pairs = list(zip(pd_gdula1.NamePhp, pd_gdula1.Partner0Php)) + list(zip(pd_gdula1.NamePhp, pd_gdula1.Partner1Php))

sin,mar = [],[]

for j in pairs:
    if j[1] == "":
        sin.append(j)
    else:
        mar.append(j)
        


# In[177]:


all = list(set(tuple_people(sin) + tuple_people(mar)))
all_count = {}
ac = 0

#unwanted_num = {11, 5} 
  
#list1 = [ele for ele in list1 if ele not in unwanted_num] 

for j in all:
    to2 = []
    unwant = {j}
    t = list(filter(lambda x:j in x, pairs))
    curr_m = tuple_people(t)
    if len(curr_m) >= 1:
        [to2.append(n) for n in curr_m]
        for m in curr_m:
            unwant.add(m)
            s = list(filter(lambda x:m in x, pairs))                
            new_t2 = [e for e in tuple_people(s) if e not in unwant]
            if len(new_t2) >= 1:
                [to2.append(n) for n in new_t2]
                for q in new_t2:
                    unwant.add(q)
                    u = list(filter(lambda x:q in x, pairs))
                    new_t3 = [e for e in tuple_people(u) if e not in unwant]
                    [to2.append(x) for x in new_t3]
            else:
                [to2.append(x) for x in new_t2]

    to2.append(j)
    to2 = list(set(to2))
    to2.sort()
    all_count[ac] = to2
    ac+=1
    #count_l.append(len(to2))

res = {}
for key,value in all_count.items():
    if value not in res.values():
        res[key] = value
        
res_c = 0
result = {}
te = list(res.values())
for i in te:
    result[res_c] = i
    res_c +=1
    
#print(len(result))


# In[178]:


# double check that no one is left
a_res = []
for k in result.keys():
    v = tuple(result[k])
    a_res.append(v)
    
#print(len(tuple_people(a_res)))


# In[179]:


def create_rev(r_d):
    nir = {}
    for i in r_d.keys():
        for j in r_d[i]:
            if j not in nir.keys():
                nir[j] = [i]
            else:
                if r_d[nir[j][0]].sort()==r_d[i].sort() == True:
                    continue
    return nir


# In[180]:


# Set up edges
edges = []
node_id_rev = create_rev(result)

for j in result.keys():
    curr = result[j]
    for a in curr:
        ca = pd_gdula1.loc[pd_gdula1['NamePhp'] == a].values.tolist()
        lm = ca[0]
        jk = [i for i in range(len(lm)) if len(lm[i]) > 0]
        if 6 in jk:
            print(lm[6])
            d = node_id_rev[ca[0][6]]
            edges.append((d[0],j))
        if 7 in jk:
            d = node_id_rev[ca[0][7]]
            edges.append((d[0],j))
        if 8 in jk:
            d = node_id_rev[ca[0][8]]
            edges.append((d[0],j))
        if 9 in jk:
            d = node_id_rev[ca[0][9]]
            edges.append((d[0],j))    


# In[ ]:


# Setting up node information
node_names = {}
hv = []
for j in result.keys():
    nw = ''
    ent = result[j]
    for k in range(len(ent)):
        f = pd_gdula1.loc[pd_gdula1['NamePhp'] == ent[k]].values.tolist()
        if k != len(ent)-1:
            nw+=f[0][2]+'; '
        else:
            nw+=f[0][2]+' '+str(j)
            
    node_names[j] = nw
    hv.append(nw)


# In[ ]:


def get_person_info(id):
    val = pd_gdula1[pd_gdula1['NamePhp']==id].index.tolist()
    l = pd_gdula1.iloc[val[0],]
    k = l.keys()
    la = [x for x in range(len(l)) if l[x] != '']
    ka = [k[x] for x in la]
    
    ts = "<b>"+l['Name']+"</b>"
    if 'Birth' in ka:
        ts+="<br>Birth: "+l['Birth']
    if 'Death' in ka:
        ts+="<br>Death: "+l['Death']
    if 'SiblingsPhp' in ka:
        ts+="<br># of Sibs: "+str(len(l['SiblingsPhp']))
    ts+='<br><br>'#+str(no)+'<br>'
    return ts


# In[ ]:


def make_hovtext(tree_dict,f_dict):
    hov = []
    if isinstance(tree_dict,dict)== True:
        l = list(tree_dict.keys())
    else:
        l = tree_dict
    for j in l:
        nm_st = ''
        for n in f_dict[j]:
            nm_st+=get_person_info(n)
        nm_st=nm_st[:-4]#+=str(j)
        #nm_st+='<extra></extra>'
        hov.append(nm_st)
    return hov


# In[ ]:


test_fam = nx.DiGraph()

[test_fam.add_node(j) for j in result.keys()]

test_fam.add_edges_from(edges)


# In[181]:


def make_edges(net,vis):
    pos=graphviz_layout(net, prog='dot')
    edge_x = []
    edge_y = []
    
    for edge in net.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    return go.Scatter(
        x=edge_x, y=edge_y,
        visible=vis,
        line=dict(width=0.5, color='#888'),
        #hoverinfo='none',
        mode='lines')


# In[182]:


def make_nodes(net,vis):
    pos=graphviz_layout(net, prog='dot')
    node_x,node_y = [],[]
    for node in net.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    return go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        visible=vis,
        hoverinfo='text',
        marker=dict(
            color=[],
            size=10,
        line_width=2))


# In[183]:


def make_node_adj(net):
    node_adjacencies = []
    for node, adjacencies in enumerate(net.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    return node_adjacencies


# ## Testing subplots

# In[184]:


def get_subnode(stn):
    if stn == -1:
        return []
    p,k = list(test_fam.predecessors(stn)),list(test_fam.successors(stn))
    l = [list(test_fam.successors(m)) for m in k]
    la = []
    for mn in l:
        [la.append(j) for j in mn]
    gen = [stn]+p+k+la
    return gen


# In[185]:


def make_subplot(test_node,o_net,net_dict,vis):
    if vis == False:
        k = o_net.subgraph(get_subnode(1))
        l = list(k.nodes())
    else:
        k = o_net.subgraph(get_subnode(test_node))
        l = list(k.nodes())
    
    subedge_trace = make_edges(k,vis)
    subnode_trace = make_nodes(k,vis)
    subnode_trace.marker.color = make_node_adj(k)
    
    if vis == True:
        hov1 = []
        for j in k:
            nm_st = ''
            for n in net_dict[j]:
                nm_st+=get_person_info(n)
            #nm_st+=str(j)
        hov1.append(nm_st)
        subnode_trace.text = hov1

    
    '''sub_t = go.Figure(data=[subnode_trace, subedge_trace],
                layout=go.Layout(
                title='Family Tree',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=20,r=30,t=40),
                annotations=[ dict(
                    text="Family tree info: <a href='http://www.gdula.info/'> http://www.gdula.info/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))'''
                              
    orig_i = l.index(test_node)
    colors = ['#05b8cc'] * len(l)
    colors[orig_i] = '#71eeb8'
    subnode_trace.marker.color = colors
    size = [10] * len(l)
    size[orig_i] = 20
    subnode_trace.marker.size = size

    return subedge_trace, subnode_trace


# In[186]:


def sub_title(intext,num):
    tit = intext[num].split('b>')
    sub_n=tit[1::2]
    sub_n1=[x[:-2] for x in sub_n]
    out = ''
    out1=[sub_n1[i]+',' if (i < len(sub_n1)-3)  else sub_n1[i]+' and ' if i < len(sub_n1)-1 else sub_n1[i] for i in range(len(sub_n1))]
    return out.join(out1)


# In[187]:


from plotly.subplots import make_subplots
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import widgets

fig = go.FigureWidget(make_subplots(rows=1, cols=2,column_widths=[0.7, 0.3]))

edge_trace = make_edges(test_fam,True)
node_trace = make_nodes(test_fam,True)

nt_text = make_hovtext(result,result)
v,w = make_subplot(1,test_fam,result,False)
#print(v.x)
fig.add_scatter(x=edge_trace.x,y=edge_trace.y,mode='lines',row=1,col=1)
fig.add_scatter(x=node_trace.x,y=node_trace.y,mode='markers',text=nt_text,row=1,col=1)
fig.add_scatter(x=v.x,y=v.y,visible=False,row=1,col=2)
fig.add_scatter(x=w.x,y=w.y,mode='markers',visible=False,row=1,col=2)

node_adjacencies = make_node_adj(test_fam)

node_trace.marker.color = node_adjacencies
node_trace.text = make_hovtext(result,result)

#print(fig)
pos3=graphviz_layout(test_fam, prog='dot')
# edge_trace
edge = fig.data[0]
edge.line.color=edge_trace.line.color
edge.line.width=edge_trace.line.width


scatter = fig.data[1]
colors = ['#05b8cc'] * len(pos3)
scatter.marker.color = colors
scatter.marker.size = [10] * len(pos3)
scatter.hovertext=nt_text
scatter.hoverinfo="text"
#print(t1.layout)#fig.data[0],'\n',edge_trace)

scatt_v = fig.data[2]
scatt_w = fig.data[3]
#fig.add_trace(t1,row=1,col=1)

today = date.today()
d1 = today.strftime("%d/%m/%Y")
#print(type(d1))
sub_note = "Family tree info as of "+str(d1)+": <a href='http://www.gdula.info/'> http://www.gdula.info/</a>"
s_note = dict(text=sub_note,
              showarrow=False,
              xref="paper", yref="paper",
              x=0.005, y=-0.002 ) 
title1 = {'font': {'size': 14},
          'showarrow': False,
          'text': 'Entire Tree',
          'x': 0.045,
          'xanchor': 'center',
          'xref': 'paper',
          'y': 1.0,
          'yanchor': 'bottom',
          'yref': 'paper'}
title2 = {'font': {'size': 14},
          'showarrow': False,
          'text': 'Family of:',
          'x': 0.775,
          'xanchor': 'center',
          'xref': 'paper',
          'y': 1.0,
          'yanchor': 'bottom',
          'yref': 'paper'}

fig.layout.hovermode = 'closest'
fig.layout.showlegend = False
fig.layout.annotations = [s_note,title1,title2]
fig.layout.margin =dict(b=20,l=20,r=30,t=40)
#print(fig.layout)
fig.layout.xaxis.update({'showgrid':False, 'zeroline':False, 'showticklabels':False})
fig.layout.xaxis2.update({'showgrid':False, 'zeroline':False, 'showticklabels':False})
fig.layout.yaxis.update({'showgrid':False, 'zeroline':False, 'showticklabels':False})
fig.layout.yaxis2.update({'showgrid':False, 'zeroline':False, 'showticklabels':False})
#fig.layout.height = 700

fig.layout.title = {'text':'Gdula Family Tree', 'font_size':16}
# Update title and height
#fig.update_layout(title_text="Gdula Family Tree", height=700)
#print(fig.layout)
#[print(type(i)) for i in fig.layout.annotations]

# create our callback function
def update_point(trace, points, selector):
    #print(selector)
    c = list(scatter.marker.color)
    s = list(scatter.marker.size)

    nv,nw = scatt_v,scatt_w
    #print(points)
    t = {}
    vis = False
    st = 'Family of: '
    for i in points.point_inds:
        #print(i)
        if c[i] == '#05b8cc':
            #print(nt_text[i])
            c[i],s[i],t[i] = '#71eeb8',20,'1'
            vis=True
            nv,nw = make_subplot(i,test_fam,result,vis)
            st += sub_title(nt_text,i)
            #print(st)
        elif c[i] == '#71eeb8':
            #print(i)
            c[i],s[i],t[i] = '#05b8cc',10,0
            vis = False
            nv,nw = make_subplot(i,test_fam,result,vis)
            #st = 'Family of:'
    for j in range(len(c)):
        if j not in t.keys():
            c[j],s[j] = '#05b8cc',10
            #vis = False
    with fig.batch_update():
        scatter.marker.color = c
        scatter.marker.size = s
        #print(nv)
        scatt_v.x,scatt_v.y = nv.x,nv.y
        #scatt_v.mode='lines'
        scatt_v.line.color=nv.line.color
        scatt_v.line.width=nv.line.width
        scatt_v.visible=vis
        scatt_v.marker.color=nv.marker.color
        scatt_w.x,scatt_w.y = nw.x,nw.y
        scatt_w.marker.color=nw.marker.color
        scatt_w.visible=vis
        title2['text']=st
        fig.layout.annotations = [s_note,title1,title2]
        scatt_w.hovertext=nw.text
        scatt_w.hoverinfo="text"

scatter.on_click(update_point)

#fig#.show()
print(fig)
#print(fig)


# In[188]:


end_time = time.time()

print('Overall time to currently run code is: ',end_time-start_time,' seconds; or ',(end_time-start_time)/60,' minutes')


# ## Testing/debugging

# In[189]:


import chart_studio.tools as tls

#tls.get_embed('https://plotly.com/~chris/1638')


# In[190]:


'''import plotly.express as px

#fig =px.scatter(x=range(10), y=range(10))
fig.write_html("Belated_birthday_present.html")'''


# # Examples of networkx plots

# # Example of plotly Networks plots

# In[ ]:





# In[ ]:




