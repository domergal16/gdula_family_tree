import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from bs4 import BeautifulSoup
import csv
import datetime
from datetime import date
import html2text
import json
import math
from matplotlib import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pd.options.mode.chained_assignment = None 
import re
import requests
import time
import urllib
import urllib.request

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

## Gets the most recent updates from the site
def curr_update():
    x2 = requests.get('http://www.gdula.info/272-2')
    s1 = BeautifulSoup(x2.content, 'html.parser')
    pub_date = s1.find_all('meta',{"property":"article:modified_time"})
    new_date = pub_date[0]['content']

    nday = int(new_date[8:10])
    nmonth = int(new_date[5:7])
    nyear = int(new_date[2:4])

    return str(nmonth)+'-'+str(nday)+'-'+str(nyear)

curr_time = curr_update()

all_keys = ['NamePhp','Name','Birth','Death','Father','Mother','Stepfather','Stepmother','Siblings','Partner0','Marriage','Partner1','Marriage License','Cause Of Death']

def get_html_info(URL):
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')
    td = soup.find_all('tr')
    out_dict = {}
    out_dict['NamePhp'] = URL.replace('http://www.gdula.info/genealogy-'+curr_time+'/','')
    sib_ct=0
    part_ct=0
    for j in td:
        ent = j.find_all(class_="ColumnAttribute")
        if len(ent) > 0:   
            Ent = ent[0].get_text().replace('\r','').replace('\n','')
            if 'Birth Name' in Ent or '_BIRTNM' in Ent:
                val2 = j.find_all(class_="ColumnValue")[0].get_text().replace('\r','').replace('\n','').replace('\t','')
                #print(val2)
                if len(val2) > 3 and ', ' in val2:
                    val2a = val2.split(', ')
                    out_dict['Name'] = val2a[1]+' '+val2a[0]
                else:
                    out_dict['Name'] = val2
            if 'father' in ent[0].get_text() or 'mother' in ent[0].get_text().lower():
                val = j.find_all('a')
                out_dict[Ent] = val[0]['href'].replace('../','')
            if 'Wife' in ent[0].get_text() or 'Husband' in ent[0].get_text() or 'Spouse' in ent[0].get_text() or 'Partner' in ent[0].get_text():
                val = j.find_all('a')
                part_ct +=1
                if part_ct > 1:
                    out_dict['Partner1'] = val[0]['href'].replace('../','')
                else:
                    out_dict['Partner0'] = val[0]['href'].replace('../','')
            if 'rother' in ent[0].get_text() or 'ister' in ent[0].get_text():
                sib_ct+=1
                out_dict['Siblings'] = str(sib_ct)
        
        ent1 = j.find_all(class_="ColumnEvent")
        if len(ent1)>0 and ('Birth' in ent1[0].get_text() or 'Death' in ent1[0].get_text() or 'Marriage' in ent1[0].get_text()):
            Ent1 = ent1[0].get_text().replace('\r','').replace('\n','').replace('\t','')
            val1a = j.find_all(class_="ColumnDate")
            val2a = j.find_all(class_="ColumnPlace")
            outs = ''
            if Ent1 not in out_dict.keys():
                if len(val1a) > 0 and val1a[0].get_text() != '\xa0':
                        outs += val1a[0].get_text().replace('\t','').replace('\r','').replace('\n','')
                if len(val2a) > 0 and val2a[0].get_text() != '\xa0':
                        outs += ', '+val2a[0].get_text().replace('\t','').replace('\r','').replace('\n','')
                out_dict[Ent1] = outs
            else:
                if 'Marriage' in ent1[0].get_text():
                    nout = [out_dict[Ent1]]
                    if len(val1a) > 0 and val1a[0].get_text() != '\xa0':
                        outs += val1a[0].get_text().replace('\t','').replace('\r','').replace('\n','')
                    if len(val2a) > 0 and val2a[0].get_text() != '\xa0':
                        outs += ', '+val2a[0].get_text().replace('\t','').replace('\r','').replace('\n','')
                    nout.append(outs)
                    out_dict[Ent1] = nout

    for i in all_keys:
        if i not in out_dict.keys():
            out_dict[i] = ""
    
    return out_dict


def get_url_paths(url, ext='', params={}):
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent

def make_dataframe(all_keys):
    #all_results = []
    url = 'http://www.gdula.info/genealogy-'+curr_time+'/ppl/'
    ext1 = '/'
    app = ['ppl//']
    result1 = [url for url in get_url_paths(url, ext1) if any(sub not in url for sub in app)]
    result2 = [x for i in result1 for x in get_url_paths(i, ext1) if len(x) == len(url)+4]
    ext2 = 'php'
    result3 = [y for j in result2 for y in get_url_paths(j,ext2)]
    
    end_res = [url for url in result3 if any(sub not in url for sub in result2)]
    
    df = pd.DataFrame(columns=all_keys)
    for t in end_res:
        df = df.append(get_html_info(t), ignore_index=True)

    return df.replace(regex=r'.php', value='')

# ## Merge partners into one node

def tuple_people(pr):
    tot = []
    for j in pr:
        tot.append(j[0])
        n = len(j)
        for k in range(1,n):
            if j[k] != "":
                tot.append(j[k])
        
    return list(set(tot))

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

def check_df(pa):
    ke = list(pa.columns)

    pa.iloc[:,:].fillna("",inplace=True)

    for k in ke:
        li = list(pa[k])
        for j in range(len(li)):
            if isinstance(li[j],float) == True or isinstance(li[j],int) == True:
                li[j] = str(int(li[j]))
                #print(k,'\tli[j] = ',li[j],' ',type(li[j]))
            if isinstance(li[j],list) == True:
                m = [x for x in li[j] if x != '']
                if len(m) == 1:
                    li[j] = m[0]
                else:
                    r = '; '.join(m)
                    li[j] = r
        #print('li = ',li)
        pa[k] = li

    new_k = list(pa.columns)
    added = [new_k.index(x) for x in new_k if x not in all_keys]
    for i in reversed(added):
        pa.drop(pa.columns[i], axis=1, inplace=True)

    pa.replace(np.nan, '', regex=True)
    return pa

def make_family_nodes(pa):
    pa = check_df(pa)
    pairs = list(zip(pa.NamePhp, pa.Partner0)) + list(zip(pa.NamePhp, pa.Partner1))
    sin,mar = [],[]

    [sin.append(j) if j[1] == "" else mar.append(j) for j in pairs]
    
    all = list(set(tuple_people(sin) + tuple_people(mar)))
    all_count,ac = {},0

    for j in all:
        to2,unwant = [],{j}
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
        to2a = [x for x in to2 if str(x) != 'nan']
        to2a = list(set(to2a))
        #print('to2 = ',to2a)
        to2a.sort()
        all_count[ac] = to2a
        ac+=1
        #count_l.append(len(to2))

    res = {}
    for key,value in all_count.items():
        if value not in res.values():
            res[key] = value
        
    res_c,result = 0,{}
    for i in list(res.values()):
        result[res_c] = i
        res_c +=1
    
    # double check that no one is left
    a_res = []
    for k in result.keys():
        v = tuple(result[k])
        a_res.append(v)

    pa = check_df(pa)
    return pa, result

def exist_files():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        if 'Gdula' in f and 'nodes_info' in f:
            c = f.split('_')
            return c[1]
    return 'none'

def file_comp():
    curr_d = curr_update()
    exist_d = exist_files()
    if curr_d == exist_d:
        return 'Saved file dates match latest files from http://gdula.info'
    else:
        return 'http://gdula.info has been updated since files have been saved. Please re-run the entire script and generate new files.'

# Define pd_gdula1
curr_path=os.getcwd()
c_file = 'Gdula_'+curr_time+'_all_info.csv'
c_e_file = 'Gdula_'+curr_time+'_nodes_info.csv'
c_r_file = 'Gdula_'+curr_time+'_results_info.txt'
csv_o_file = os.path.join(curr_path, c_file) # individual family members
csv_n_file = os.path.join(curr_path, c_e_file) # stored as nodes
txt_r_file = os.path.join(curr_path,c_r_file) # results dictionary

if os.path.isfile(csv_n_file) and os.path.isfile(txt_r_file): # if nodes csv is saved
    pd_gdula1 = pd.read_csv(csv_n_file)
    with open(txt_r_file, 'rb') as handle:
        result = pickle.loads(handle.read())
        
elif os.path.isfile(csv_o_file): # if ind. csv is saved
    pd_gdula = pd.read_csv(csv_o_file)
    pd_gdula.replace(np.nan, '', regex=True)
    pd_gdula1,result = make_family_nodes(pd_gdula)
    pd_gdula1.to_csv(csv_n_file,header=True)
    with open(txt_r_file, 'wb') as handle:
        pickle.dump(result, handle)

elif curr_time != exist_files():# if website has been updated 
    n_c_t = exist_files()
    c_ne_file = 'Gdula_'+n_c_t+'_nodes_info.csv'
    c_nr_file = 'Gdula_'+n_c_t+'_results_info.txt'
    csv_nn_file = os.path.join(curr_path, c_e_file) # stored as nodes
    txt_nr_file = os.path.join(curr_path,c_r_file) # results dictionary
    pd_gdula1 = pd.read_csv(csv_nn_file)
    with open(txt_nr_file, 'rb') as handle:
        result = pickle.loads(handle.read())
    
else: # if neither is saved
    pd_gdula = make_dataframe(all_keys)
    pd_gdula.to_csv(csv_o_file, header=True)
    pd_gdula.replace(np.nan, '', regex=True)
    pd_gdula1,result = make_family_nodes(pd_gdula)
    pd_gdula1.to_csv(csv_n_file, header=True)
    with open(txt_r_file, 'wb') as handle:
        pickle.dump(result, handle)

pd_gdula1.iloc[:,:].fillna("",inplace=True)
pd_gdula1 = check_df(pd_gdula1)
#pd_gdula1
#print(len(result))

#print(pd_gdula1.columns)
## Nodes and Edges

# Set up edges
edges = []
node_id_rev = create_rev(result)

for j in result.keys():
    curr = result[j]
    for a in curr:
        ca = pd_gdula1.loc[pd_gdula1['NamePhp'] == a].values.tolist()
        jk = [i for i in range(1,len(ca[0])) if len(ca[0][i]) > 0]
        if pd_gdula1.columns.get_loc("Father") in jk:
            d = node_id_rev[ca[0][pd_gdula1.columns.get_loc("Father")]]
            edges.append((d[0],j))
        if pd_gdula1.columns.get_loc("Stepfather") in jk:
            d = node_id_rev[ca[0][pd_gdula1.columns.get_loc("Stepfather")]]
            edges.append((d[0],j))
        if pd_gdula1.columns.get_loc("Mother") in jk:
            d = node_id_rev[ca[0][pd_gdula1.columns.get_loc("Mother")]]
            edges.append((d[0],j))
        if pd_gdula1.columns.get_loc("Stepmother") in jk:
            d = node_id_rev[ca[0][pd_gdula1.columns.get_loc("Stepmother")]]
            edges.append((d[0],j))     

# Setting up node information
node_names = {}
hv = []
for j in result.keys():
    nw = ''
    ent = result[j]
    for k in range(len(ent)):
        f = pd_gdula1.loc[pd_gdula1['NamePhp'] == ent[k]].values.tolist()
        # print('f = ',f)
        if k != len(ent)-1:
            if isinstance(f[0][2],list):
                nw+=f[0][2][0]+'; '
            else:
                nw+=f[0][2]+'; '
        else:
            nw+=f[0][2]+' '+str(j)
            
    node_names[j] = nw
    hv.append(nw)

def get_person_info(id):
    val = pd_gdula1[pd_gdula1['NamePhp']==id].index.tolist()
    l = pd_gdula1.iloc[val[0],]
    k = l.keys()
    la = [x for x in range(len(l)) if l[x] != '']
    ka = [k[x] for x in la]
    
    ts = "<b>"+l['Name']+"</b>"
    if 'Birth' in ka:
        ts+="<br>Birth: "+l['Birth']
    if 'Marriage' in ka:
        ts+="<br>Marriage: "+l['Marriage']
    if 'Marriage License' in ka:
        ts+="<br>Marriage License: "+l['Marriage License']
    if 'Death' in ka:
        ts+="<br>Death: "+l['Death']
    if 'Cause of Death' in ka:
        ts+="<br>Cause of Death: "+l['Cause of Death']
    if 'SiblingsPhp' in ka:
        ts+="<br># of Sibs: "+l['Siblings']
    ts+='<br><br>'#+str(no)+'<br>'
    return ts

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

test_fam = nx.DiGraph()
[test_fam.add_node(j) for j in result.keys()]
test_fam.add_edges_from(edges)

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

def make_nodes(net,vis):
    print('net = ',net)
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

def make_node_adj(net):
    node_adjacencies = []
    for node, adjacencies in enumerate(net.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    return node_adjacencies

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

def make_subplot(test_node,o_net,net_dict,vis):
    if vis == False:
        k = o_net.subgraph(get_subnode(1))
        l = list(k.nodes())
    else:
        k = o_net.subgraph(get_subnode(test_node))
        l = list(k.nodes())
    print#('test_node=',test_node)
    
    #print('k = ',k)
    subedge_trace = make_edges(k,vis)
    subnode_trace = make_nodes(k,vis)
    #subnode_trace.marker.color = make_node_adj(k)
    
    if vis == True:
        hov1 = make_hovtext(l,net_dict)
        subnode_trace.text = hov1
        subnode_trace.hovertext=hov1
        subnode_trace.hoverinfo="text"
                              
    orig_i = l.index(test_node)
    colors = ['#05b8cc'] * len(l)
    colors[orig_i] = '#71eeb8'
    subnode_trace.marker.color = colors
    size = [10] * len(l)
    size[orig_i] = 20
    subnode_trace.marker.size = size

    print(len(l))
    print('colors = ',colors)

    return subedge_trace, subnode_trace

def sub_title(intext,num):
    tit = intext[num].split('b>')
    sub_n=tit[1::2]
    sub_n1=[x[:-2] for x in sub_n]
    out = ''
    out1=[sub_n1[i]+',' if (i < len(sub_n1)-3)  else sub_n1[i]+' and ' if i < len(sub_n1)-1 else sub_n1[i] for i in range(len(sub_n1))]
    return out.join(out1)

## FIG ##

def make_fig():#snode,vis):
    fig = go.FigureWidget(make_subplots(rows=1, cols=2,column_widths=[0.7, 0.3]))
    edge_trace = make_edges(test_fam,True)
    node_trace = make_nodes(test_fam,True)

    nt_text = make_hovtext(result,result)
    v,w = make_subplot(1,test_fam,result,False)

    fig.add_scatter(x=edge_trace.x,y=edge_trace.y,mode='lines',row=1,col=1)
    fig.add_scatter(x=node_trace.x,y=node_trace.y,mode='markers',text=nt_text,row=1,col=1)
    fig.add_scatter(x=v.x,y=v.y,visible=v.visible,row=1,col=2)
    fig.add_scatter(x=w.x,y=w.y,mode='markers',visible=w.visible,row=1,col=2)

    node_adjacencies = make_node_adj(test_fam)

    #node_trace.marker.color = node_adjacencies
    node_trace.text = make_hovtext(result,result)

    pos3=graphviz_layout(test_fam, prog='dot')

    # edge_trace
    edge,scatter = fig.data[0],fig.data[1]
    edge.line.color,edge.line.width=edge_trace.line.color,edge_trace.line.width

    scatter.marker.color,scatter.marker.size = ['#05b8cc'] * len(pos3),[10]*len(pos3)
    scatter.hovertext,scatter.hoverinfo=make_hovtext(result,result),"text"

    scatt_v,scatt_w = fig.data[2],fig.data[3]
    scatt_v.mode='lines'
    scatt_v.line.color,scatt_v.line.width=v.line.color,v.line.width
    scatt_w.marker.size,scatt_w.marker.color = w.marker.size,w.marker.color

    today = date.today()
    d1 = today.strftime("%d/%m/%Y")

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
    fig.layout.clickmode = 'event+select'
    fig.layout.annotations = [s_note,title1,title2]
    fig.layout.margin =dict(b=20,l=20,r=30,t=40)

    fig.layout.xaxis.update({'showgrid':False, 'zeroline':False, 'showticklabels':False})
    fig.layout.xaxis2.update({'showgrid':False, 'zeroline':False, 'showticklabels':False})
    fig.layout.yaxis.update({'showgrid':False, 'zeroline':False, 'showticklabels':False})
    fig.layout.yaxis2.update({'showgrid':False, 'zeroline':False, 'showticklabels':False})

    fig.layout.title = {'text':'Gdula Family Tree', 'font_size':16} 

    return fig

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server=app.server

app.layout = html.Div(children=[
    html.H1(children='Gdula Family Tree'),

    html.Div(children='''
        Family Tree code written by Elizabeth Sudkamp. Information pulled from gdula.info . Written using NetworkX, Plotly, and Dash.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=make_fig()
    ),
    html.Br(),
    html.Div(id='my-output',children='File status: {}'.format(file_comp())
),
])

@app.callback(
    Output('example-graph', 'figure'),
    [Input('example-graph', 'clickData')])
# create our callback function
def update_point(selectInput):
    #print('select Input = ',selectInput)
    nfig = make_fig()
    #scatter,scatt_v,scatt_w = nfig.data[1],nfig.data[2],nfig.data[3]
    title2 = nfig.layout.annotations[2]
    
    c = list(nfig.data[1].marker.color)
    s = list(nfig.data[1].marker.size)

    nv,nw = scatt_v,scatt_w
    
    t,vis,st = {},False,'Family of: ' 
    try:
        print(selectInput['points'])
    except TypeError:
        return nfig
    else:
        pt = selectInput['points']
        for im in range(len(selectInput['points'])):
            i = selectInput['points'][im]['pointIndex']
            if c[i] == '#05b8cc':
                c[i],s[i],t[i],vis = '#71eeb8',20,'1',True
                nv,nw = make_subplot(i,test_fam,result,vis)
                st += sub_title(scatter.hovertext,i)
            elif c[i] == '#71eeb8':
                c[i],s[i],t[i],vis = '#05b8cc',10,0,False
                nv,nw = make_subplot(1,test_fam,result,vis)
        for j in range(len(c)):
            if j not in t.keys():
                c[j],s[j] = '#05b8cc',10
        #with nfig.batch_update():
        nfig.data[1].marker.color,nfig.data[1].marker.size = c,s
        scatt_v.x,scatt_v.y = nv.x,nv.y
        scatt_v.mode='lines'
        scatt_v.line.color,scatt_v.line.width,scatt_v.visible=nv.line.color,nv.line.width,vis
        scatt_w.x,scatt_w.y = nw.x,nw.y
        print('out_marker colors: ',nw.marker.color)
        scatt_w.marker.color,scatt_w.visible=nw.marker.color,vis
        title2['text']=st
        ann = list(nfig.layout.annotations)
        ann[2] = title2
        nfig.layout.annotations = tuple(ann)
        scatt_w.hovertext,scatt_w.hoverinfo=nw.text,"text"
        #nfig.data[1],nfig.data[2],nfig.data[3] = scatter,scatt_v,scatt_w
        
        return nfig


if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=True)
