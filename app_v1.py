import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from flask import Flask, g, make_response, request
from flask_caching import Cache

import boto3
from bs4 import BeautifulSoup
import csv
import datetime
from datetime import date
import graphviz
import html2text
import io
import json
import math
from matplotlib import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pd.options.mode.chained_assignment = None 
import pydot_ng as pydot
import re
import requests
from s3fs.core import S3FileSystem
import time
import urllib
import urllib.request

#### APP and Server setup #####

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server=app.server

app.title = 'Gdula Family Tree'
app.config.suppress_callback_exceptions = True
########

#### FUNCTIONS

### Initial Functions (Getting data frame set up/read in/etc)

## Gets the most recent updates from the site
def curr_update():
    x2 = requests.get('http://www.gdula.info/272-2')
    s1 = BeautifulSoup(x2.content, 'html.parser')
    pub_date = s1.find_all('meta',{"property":"article:modified_time"})
    new_date = pub_date[0]['content']

    return str(int(new_date[5:7]))+'-'+str(int(new_date[8:10]))+'-'+str(int(new_date[2:4]))

def get_html_info(URL,curr_time,a_keys):
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')
    td = soup.find_all('tr')
    out_dict = {}
    out_dict['NamePhp'] = URL.replace('http://www.gdula.info/genealogy-'+curr_time+'/','')
    sib_ct,part_ct=0,0
    for j in td:
        ent = j.find_all(class_="ColumnAttribute")
        if len(ent) > 0:   
            Ent = ent[0].get_text().replace('\r','').replace('\n','')
            if 'Birth Name' in Ent or '_BIRTNM' in Ent:
                val2 = j.find_all(class_="ColumnValue")[0].get_text().replace('\r','').replace('\n','').replace('\t','')
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
            val1a,val2a,outs = j.find_all(class_="ColumnDate"),j.find_all(class_="ColumnPlace"),''
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

    for i in a_keys:
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

def make_dataframe(all_keys,curr_time,a_k):
    #all_results = []
    url = 'http://www.gdula.info/genealogy-'+curr_time+'/ppl/'
    ext1,ext2 = '/','php'
    app = ['ppl//']
    result1 = [url for url in get_url_paths(url, ext1) if any(sub not in url for sub in app)]
    result2 = [x for i in result1 for x in get_url_paths(i, ext1) if len(x) == len(url)+4]
    result3 = [y for j in result2 for y in get_url_paths(j,ext2)]   
    end_res = [url for url in result3 if any(sub not in url for sub in result2)]
    
    df = pd.DataFrame(columns=all_keys)
    for t in end_res:
        df = df.append(get_html_info(t,curr_time,a_k), ignore_index=True)

    return df.replace(regex=r'.php', value='')

# ## Merge partners into one node

def tuple_people(pr):
    tot = []
    for j in pr:
        tot.append(j[0])
        for k in range(1,len(j)):
            if j[k] != "":
                tot.append(j[k])
        
    return list(set(tot))

def create_rev(r_d):
    nir = {}
    for i in r_d.keys():
        l = r_d[i]
        for j in l:
            if j not in nir.keys():
                nir[j] = [i]
            else:
                if r_d[nir[j][0]].sort()==r_d[i].sort() == True:
                    continue
    return nir

def check_df(pa,all_keys):
    ke = list(pa.columns)

    pa.iloc[:,:].fillna("",inplace=True)

    for k in ke:
        li = list(pa[k])
        for j in range(len(li)):
            if isinstance(li[j],float) == True or isinstance(li[j],int) == True:
                li[j] = str(int(li[j]))
            if isinstance(li[j],list) == True:
                m = [x for x in li[j] if x != '']
                if len(m) == 1:
                    li[j] = m[0]
                else:
                    li[j] = '; '.join(m)
        pa[k] = li

    added = [list(pa.columns).index(x) for x in list(pa.columns) if x not in all_keys]
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
        to2a.sort()
        all_count[ac] = to2a
        ac+=1

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
        a_res.append(tuple(result[k]))

    pa = check_df(pa)
    return pa, result

def exist_files():
    s3 = boto3.resource('s3')
    bucket_name = 'gdula-fam-files'
    my_bucket = s3.Bucket(bucket_name)
    
    for file in my_bucket.objects.all():
        if 'Gdula' in file.key and 'nodes_info' in file.key:
            c = file.key.split('_')
            return c[1]
    return 'none'

def file_comp():
    if curr_update() == exist_files():
        return 'Saved file dates match latest files from http://gdula.info'
    else:
        curr_time = curr_update()

        # Define pd_gdula1
        c_file,c_e_file,c_r_file = 'Gdula_'+curr_time+'_all_info.csv','Gdula_'+curr_time+'_nodes_info.csv','Gdula_'+curr_time+'_results_info.txt'

        resources,client = boto3.resource('s3'),boto3.client('s3')

        bucket_name = 'gdula-fam-files'
        my_bucket = resources.Bucket(bucket_name)

        file_list = []
        for file in my_bucket.objects.all():
            file_list.append(file.key)
            
        if c_e_efile not in file_list or c_r_file not in file_list:
            return 'not all files have been uploaded. Please re-run entire script and upload the new files.'
        else:
            return 'http://gdula.info has been updated since files have been saved. Please re-run the entire script and generate new files.'

#### Nodes and Edges Functions

def create_edges(pa,res):
    ed,node_id_rev = [],create_rev(res)

    for j in res.keys():
        curr = res[j]
        for a in curr:
            ca = pa.loc[pa['NamePhp'] == a].values.tolist()
            jk = [i for i in range(1,len(ca[0])) if len(ca[0][i]) > 0]
            if pa.columns.get_loc("Father") in jk:
                d = node_id_rev[ca[0][pa.columns.get_loc("Father")]]
                ed.append((d[0],j))
            if pa.columns.get_loc("Stepfather") in jk:
                d = node_id_rev[ca[0][pa.columns.get_loc("Stepfather")]]
                ed.append((d[0],j))
            if pa.columns.get_loc("Mother") in jk:
                d = node_id_rev[ca[0][pa.columns.get_loc("Mother")]]
                ed.append((d[0],j))
            if pa.columns.get_loc("Stepmother") in jk:
                d = node_id_rev[ca[0][pa.columns.get_loc("Stepmother")]]
                ed.append((d[0],j))
    return ed



def make_edges_nodes(net,vis):
    pos=graphviz_layout(net, prog='dot')
    edge_x,edge_y = [],[]
    
    for edge in net.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
    node_x,node_y = [],[]
    for node in net.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    return go.Scatter(
        x=edge_x, y=edge_y,
        visible=vis,
        uid='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'),go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        visible=vis,
        uid='nodes',
        hoverinfo='text',
        marker=dict(
            color=[],
            size=10,
        line_width=2))


def get_subnode(stn,t_f):
    p,k = list(t_f.predecessors(stn)),list(t_f.successors(stn))
    la = []
    for mn in [list(t_f.successors(m)) for m in k]:
        [la.append(j) for j in mn]
    gen = [stn]+p+k+la
    return gen

def generate_net(pa,res):
    t_f = nx.DiGraph()
    [t_f.add_node(j) for j in res.keys()]
    t_f.add_edges_from(create_edges(pa,res))
    #print('t_f is done')
    return t_f

#### Plots/Graphs Functions

def get_person_info(id,p_d):
    val = p_d[p_d['NamePhp']==id].index.tolist()
    l = p_d.iloc[val[0],]
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

def make_hovtext(tree_dict,f_dict,p_d):
    hov = []
    if isinstance(tree_dict,dict)== True:
        l = list(tree_dict.keys())
    else:
        l = tree_dict
    for j in l:
        nm_st = ''
        for n in f_dict[j]:
            nm_st+=get_person_info(n,p_d)
        nm_st=nm_st[:-4]
        hov.append(nm_st)
    return hov

def make_subplot(test_node,o_net,net_dict,vis,p_d):

    l = list(o_net.subgraph(get_subnode(test_node,o_net)).nodes())

    #print('making sub_traces')
    subedge_trace,subnode_trace = make_edges_nodes(o_net.subgraph(get_subnode(test_node,o_net)),vis)

    if vis == True:
        subnode_trace.hovertext=make_hovtext(l,net_dict,p_d)
        subnode_trace.hoverinfo="text"
        
    orig_i = l.index(test_node)
    colors = ['#05b8cc'] * len(l)
    colors[orig_i] = '#71eeb8'
    subnode_trace.marker.color = colors
    size = [10] * len(l)
    size[orig_i] = 20
    subnode_trace.marker.size = size

    return subedge_trace, subnode_trace

def sub_title(intext,num):
    tit = intext[num].split('b>')
    #print('tit = ',tit[1::2])
    sub_n= tit[1::2]
    sub_n1=[x[:-2] for x in sub_n]
    out = ''
    out1=[sub_n1[i]+',' if (i < len(sub_n1)-3)  else sub_n1[i]+' and ' if i < len(sub_n1)-1 else sub_n1[i] for i in range(len(sub_n1))]
    return out.join(out1)

##### FIG ######
#@cache.memoize()
def make_fig(pa,res):

    #print('fig started')

    test_fam = generate_net(pa,res)
    #print('test_fam generated')
    fig = go.FigureWidget(make_subplots(rows=1, cols=2,column_widths=[0.7, 0.3]))
    edge_trace,node_trace = make_edges_nodes(test_fam,True)#make_edges(test_fam,True),make_nodes(test_fam,True)

    v,w = make_subplot(1,test_fam,res,False,pa)
    
    fig.add_scatter(x=edge_trace.x,y=edge_trace.y,mode='lines',row=1,col=1)
    fig.add_scatter(x=node_trace.x,y=node_trace.y,mode='markers',row=1,col=1)
    fig.add_scatter(x=v.x,y=v.y,mode='lines',row=1,col=2)
    fig.add_scatter(x=w.x,y=w.y,mode='markers',row=1,col=2)

    #print('scatters added')
    #fig.data[1].text = make_hovtext(res,res)
    pos3=graphviz_layout(test_fam, prog='dot')

    # edge_trace
    fig.data[0].line.color,fig.data[0].line.width=edge_trace.line.color,edge_trace.line.width

    fig.data[1].marker.color,fig.data[1].marker.size = ['#05b8cc'] * len(pos3), [10]*len(pos3)
    fig.data[1].hovertext,fig.data[1].hoverinfo=make_hovtext(res,res,pa),"text"

    fig.data[2].line.color,fig.data[2].line.width=v.line.color,v.line.width

    fig.data[3].marker.color,fig.data[3].marker.size = w.marker.color,w.marker.size
    fig.data[3].hovertext,fig.data[3].hoverinfo=w.hovertext,"text"
    fig.data[2].visible,fig.data[3].visible = False,False

    #print('working on layout')
    s_note = dict(text="Family tree info as of "+str(date.today().strftime("%m/%d/%Y"))+": <a href='http://www.gdula.info/'> http://www.gdula.info/</a>",showarrow=False,xref="paper", yref="paper",x=0.005, y=-0.002 ) 
    title1 = {'font': {'size': 14},'showarrow': False,'text': 'Entire Tree','x': 0.025,'xanchor': 'center','xref': 'paper','y': 1.0,'yanchor': 'bottom','yref': 'paper'}
    title2 = {'font': {'size': 14},'showarrow': False,'text': 'Family of:','x': 0.775,'xanchor': 'center','xref': 'paper','y': 1.0,'yanchor': 'bottom','yref': 'paper'}
    
    fig.layout.hovermode,fig.layout.showlegend = 'closest',False
    fig.layout.clickmode,fig.layout.annotations = 'event+select',[s_note,title1,title2]
    fig.layout.margin,fig.layout.title =dict(b=20,l=20,r=30,t=40),{'text':'Gdula Family Tree', 'font_size':16}
    
    fig.layout.xaxis.showgrid,fig.layout.xaxis.zeroline,fig.layout.xaxis.showticklabels=False,False,False
    fig.layout.yaxis.showgrid,fig.layout.yaxis.zeroline,fig.layout.yaxis.showticklabels=False,False,False

    fig.layout.xaxis2.showgrid,fig.layout.xaxis2.zeroline,fig.layout.xaxis2.showticklabels=False,False,False
    fig.layout.yaxis2.showgrid,fig.layout.yaxis2.zeroline,fig.layout.yaxis2.showticklabels=False,False,False

    #print('fig done')
    return fig

###
def setup_df():
    curr_time = curr_update()
    all_keys = ['NamePhp','Name','Birth','Death','Father','Mother','Stepfather','Stepmother','Siblings','Partner0','Marriage','Partner1','Marriage License','Cause Of Death']

    # Define pd_gdula1
    c_file = 'Gdula_'+curr_time+'_all_info.csv'
    c_e_file = 'Gdula_'+curr_time+'_nodes_info.csv'
    c_r_file = 'Gdula_'+curr_time+'_results_info.txt'

    resources = boto3.resource('s3')
    client = boto3.client('s3')

    bucket_name = 'gdula-fam-files'
    my_bucket = resources.Bucket(bucket_name)
    
    file_list = []
    for file in my_bucket.objects.all():
        file_list.append(file.key)
        
    #print('bucket accessed and files listed')
    if c_e_file in file_list and c_r_file in file_list: # if nodes csv is saved
        obj1 = client.get_object(Bucket= bucket_name , Key = c_e_file)
        pd_gdula1 = pd.read_csv(io.BytesIO(obj1['Body'].read()), encoding='utf8')
        obj2 = client.get_object(Bucket= bucket_name , Key = c_r_file)
        result = pickle.loads(obj2['Body'].read())
        
    elif c_file in file_list: # if ind. csv is saved
        obj1 = client.get_object(Bucket= bucket_name , Key = c_file)
        pd_gdula = pd.read_csv(io.BytesIO(obj1['Body'].read()), encoding='utf8')
        pd_gdula.replace(np.nan, '', regex=True)
        pd_gdula1,result = make_family_nodes(pd_gdula)

        #pd_gdula1.to_csv(csv_n_file,header=True)
        #with open(txt_r_file, 'wb') as handle:
        #    pickle.dump(result, handle)

    elif curr_time != exist_files():# if website has been updated 
        n_c_t = exist_files()
        c_ne_file = 'Gdula_'+n_c_t+'_nodes_info.csv'
        c_nr_file = 'Gdula_'+n_c_t+'_results_info.txt'

        obj1 = client.get_object(Bucket= bucket_name , Key = c_ne_file)
        pd_gdula1 = pd.read_csv(io.BytesIO(obj1['Body'].read()), encoding='utf8')
        obj2 = client.get_object(Bucket= bucket_name , Key = c_nr_file)
        result = pickle.loads(obj2['Body'].read())
        
        #pd_gdula1 = pd.read_csv(csv_nn_file)
        #with open(txt_nr_file, 'rb') as handle:
        #    result = pickle.loads(handle.read())

    else: # if neither is saved
        pd_gdula = make_dataframe(all_keys,curr_time,all_keys)
        pd_gdula.to_csv(csv_o_file, header=True)
        pd_gdula.replace(np.nan, '', regex=True)
        pd_gdula1,result = make_family_nodes(pd_gdula)
        #pd_gdula1.to_csv(csv_n_file, header=True)
        #with open(txt_r_file, 'wb') as handle:
        #    pickle.dump(result, handle)

    #print('files read in')
    pd_gdula1.iloc[:,:].fillna("",inplace=True)
    pd_gdula1 = check_df(pd_gdula1,all_keys)

    #print('files returned')
    #print(pd_gdula1.head(2))
    return pd_gdula1, result

###### End of Functions ######


#### Layout #####

app.layout = html.Div(children=[
    html.H1(children='Gdula Family Tree'),
    html.H5([dcc.Markdown(children= '''Code written by Elizabeth Sudkamp. Information pulled from [gdula.info](/). 

 Written and hosted using NetworkX, Plotly, Dash, Flask and Heroku.''')]),
    html.H2(children='Warning'),
    html.P([dcc.Markdown(children= '''If a family is not loading right away, or the main graph is unable to load, that means someone else is using this app. Please be patient, and try again later.''')]),
    
    html.Br(),
    html.Div([
        #dbc.Spinner(html.Div(id="loading-output")),
        dcc.Loading(id="loading-1",type="circle",color='#05b8cc',children=html.Div(id="loading-output")),
        html.Button("Make Graph", id="loading-button",style={'color':'#7fafdf'}),
    ]),
            
    html.P(id='my-output',children='File status: {}'.format(file_comp())
    )
])

#### Callbacks


@app.callback(
    Output("loading-output", "children"), [Input("loading-button", "n_clicks")]
)
def load_output(n):
    #print('generate function called')
    if n:
        pd_gdula1, result = setup_df()
        #print('sleeping')
        #if pd_gdula1 is None:
            #print('no value for pd_gdula1')
        #elif result is None:
            #print('no value for result')
        #else:
            #print('both pdgudla1 and results have values')
        #print('files set up')
        return html.Div(id='test-output',children=[
            #html.H5(children="Output loaded {} times".format(n)),
            dcc.Graph(
                id='example-graph',
                figure=make_fig(pd_gdula1,result))
        ])
    return "Output not loaded yet"



@app.callback(
    Output('example-graph', 'figure'),
    [Input('example-graph', 'clickData')],
    [State('example-graph', 'figure')])
# create our callback function
def update_point(selectInput,nfig):
    t,vis,st = {},False,'Family of: '
    try:
        kaq = selectInput['points']
    except TypeError:
        nfig['data'][2]['visible'] = False
        nfig['data'][3]['visible'] = False
        return nfig
    else:
        pd_gdula1, result = setup_df()
        test_fam = generate_net(pd_gdula1,result)
        for im in range(len(selectInput['points'])):
            i = selectInput['points'][im]['pointIndex']
            
            if selectInput['points'][im]['marker.color'] == '#71eeb8':
                nfig['data'][1]['marker']['color'][i],nfig['data'][1]['marker']['size'][i]= '#05b8cc',10

                nfig['data'][2]['visible'],nfig['data'][3]['visible'] = False,False
                
            else:
                nfig['data'][1]['marker']['color'][i],nfig['data'][1]['marker']['size'][i] = '#71eeb8',20
                t[i],vis = '1',True
                nv,nw = make_subplot(i,test_fam,result,vis,pd_gdula1)
                st += sub_title(nfig['data'][1]['hovertext'],i)
                
        for j in range(len(nfig['data'][1]['marker']['color'])):
            if j not in t.keys():
                nfig['data'][1]['marker']['color'][j],nfig['data'][1]['marker']['size'][j] = '#05b8cc',10

        nfig['layout']['annotations'][2]['text']=st

        if vis == True:

            nfig['data'][2]['visible'],nfig['data'][3]['visible'] = True,True
 
            nfig['data'][2]['x'],nfig['data'][2]['y'] = nv.x,nv.y
            nfig['data'][2]['uid'] += '1'
            nfig['data'][3]['uid'] += '1'
            nfig['data'][3]['x'],nfig['data'][3]['y'] = nw.x,nw.y

            nfig['data'][3]['hoverinfo'],nfig['data'][3]['hovertext'] = nw.hoverinfo,nw.hovertext
            
            nfig['data'][3]['marker']['color'],nfig['data'][3]['marker']['size']=list(nw.marker.color),nw.marker.size

        return nfig


if __name__ == '__main__':
    app.run_server(debug=True)
