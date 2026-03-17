import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from plotnine import *
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score 
from sklearn.decomposition import PCA 
from scipy.spatial.distance import pdist, squareform

data= pd.read_csv('/Users/tommygillan/Documents/Unstructured/fangraphs_addwar_updated.csv')

 ##remove all the null values from 2006 and set at bat minimum
data=data[(data['Season']!=2006)&(data['AB']>=200)]

##Age Adj subtracts the levels avg age for year from actual age 
data['Avg Age']= data.groupby(['Level', 'Season'])['Age'].transform('mean')
data['Age Adj']= data['Age']-data['Avg Age']
data=data.drop(['Avg Age'],axis=1)

# Create continuous age adjusted stat multiplier
def get_cont_multiplier(adj_age, slope = 0.10, cap_min = 0.80, cap_max = 1.30):
    multiplier = 1 - (adj_age * slope)
    if multiplier > cap_max:
        multiplier = cap_max
    elif multiplier < cap_min:
        multiplier = cap_min
    return multiplier

data['Age Multiplier']=data['Age Adj'].apply(get_cont_multiplier)
age_adjusted_cols=['wRC+','ISO','BABIP','wOBA']
for col in age_adjusted_cols:
    data[col+' Adj']=data[col]*data['Age Multiplier']
data=data.drop(['wRC+','ISO','wOBA','BABIP'],axis=1)

# Convert counting stats to rates BEFORE building clustering subset
data['HR/AB'] = data['HR'] / data['AB']
data['SB/AB'] = data['SB'] / data['AB']

##keep identifiers for labeling players later
identifier_subset=data[['Name','Team','Level','Called Up','PlayerId','WAR','Age','Season','Age Adj']]

##drop dervived stats and identifier data
clustering_subset=data.drop(['Name','Team','Level','Called Up','PlayerId', 'WAR',
'AB','H','OBP','SLG','wRC', 'Cent%', 'GB%','FB%','IFFB%','Season', 'HR', 'SB',
 'G','SO','CS','HR/FB','GB/FB','Age Adj', 'Age Multiplier', 'Age','2B','3B','AVG','RBI','BB/K'],axis=1)


'''aggregate by entire career, weigh season by number of at bats divided by total career ABs
'wRC+','ISO','BABIP' also previously weighted by age and level''' 

stat_cols = clustering_subset.columns.tolist()

ab = data['AB'].values
player_ids = data['PlayerId'].values
##multiplying each season’s stat by that season’s AB.
weighted = clustering_subset[stat_cols].multiply(ab, axis=0)
weighted['PlayerId'] = player_ids
weighted['AB'] = ab
##sum every years multiplied value and divide by total at bats

career = weighted.groupby('PlayerId')[stat_cols + ['AB']].sum()
career[stat_cols] = career[stat_cols].div(career['AB'], axis=0)
career = career.drop(columns='AB').reset_index()

##career level idenifiers
career_identifiers = (
    identifier_subset.groupby('PlayerId').agg(
        Name=('Name', 'last'),           
        Called_Up=('Called Up', 'max'),  
        WAR=('WAR', 'sum'),              
        Max_Age=('Age', 'max'),          
        Max_Level=('Level', 'last'),     
        Seasons=('Season', 'count'),
        Last_Season = ('Season', 'max')     
    )
    .reset_index()
)

## create trajectory score
data = data.sort_values(['PlayerId', 'Season'])
improvement_stats = ['wRC+ Adj', 'ISO Adj', 'BB%', 'HR/AB']
decline_stats = ['K%', 'SwStr%'] 

for col in improvement_stats:
    data[col + "_Change"] = data.groupby('PlayerId')[col].diff().fillna(0)

for col in decline_stats:
    data[col + "_Change"] = (data.groupby('PlayerId')[col].shift(1) - data[col]).fillna(0)

traj_cols = []
for col in (improvement_stats + decline_stats):
    traj_cols.append(col + "_Change")

trajectory_df = data.groupby('PlayerId')[traj_cols].mean().reset_index()

traj_scaler = StandardScaler()
traj_matrix = trajectory_df[traj_cols].values
traj_scaled_values = traj_scaler.fit_transform(traj_matrix)
trajectory_df['Trajectory_Score'] = traj_scaled_values.mean(axis=1)
career = pd.merge(career, trajectory_df[['PlayerId', 'Trajectory_Score']], on='PlayerId', how='left')


##split into training and test 
train_ids = career_identifiers[career_identifiers['Last_Season'] <= 2021]['PlayerId']
test_ids  = career_identifiers[career_identifiers['Last_Season'] >  2021]['PlayerId']
career_train = career[career['PlayerId'].isin(train_ids)].reset_index(drop=True)
career_test  = career[career['PlayerId'].isin(test_ids)].reset_index(drop=True)


##scale on just training dara
X_train=career_train.drop(columns=['PlayerId'])
feature_cols = X_train.columns.tolist()
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train.values)  
X_test  = career_test.drop(columns=['PlayerId'])
X_test_scaled = scaler.transform(X_test.values)         


##find ideal K
silhouette_scores=[]

k_range=range(4,7)
for k in k_range:
    kmeans = KMeans(n_clusters=k, 
        random_state=42, 
        n_init=25,
        algorithm="lloyd" )
    cluster_labels = kmeans.fit_predict(X_train_scaled)
    
    silhouette_avg = silhouette_score(X_train_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

best_index = np.argmax(silhouette_scores)
optimal_k = k_range[best_index]

#run clustering 
kmeans = KMeans(
    n_clusters=optimal_k, 
    n_init=25,
    max_iter=100, 
    random_state=42, 
    algorithm="lloyd"   
).fit(X_train_scaled)

career_train["cluster"] = kmeans.labels_


#visualize clusters
pca = PCA(2) 
pca_data = pd.DataFrame(pca.fit_transform(X_train_scaled),columns=['PC1','PC2']) 
pca_data['cluster'] = pd.Categorical(kmeans.labels_)

pca=(
    ggplot(pca_data)+
    aes(x='PC1', y='PC2', color='cluster')+
    geom_point(alpha=0.3)+
    theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(                                              
        x="PC1",
        y="PC2",
        title="Principal Component Analysis Baseball Prospects"
    )
)
pca


## create cluster heatmap
centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=feature_cols
)

centroids["cluster"] = np.arange(kmeans.n_clusters)

centroids_melted = (
    centroids
    .melt(id_vars=["cluster"], var_name="features", value_name="values")
    .copy()
)


g_heat_1 = (
    ggplot(data=centroids_melted,                           
           mapping=aes(x="features", y="cluster", fill="values"))
    + scale_y_continuous(breaks=list(range(0, 4)))     
    + geom_tile()                                        
    + theme_bw()                             
    + scale_fill_gradient2(                              
        low="blue", mid="white", high="red",
        midpoint=0, na_value="grey"        
    )
    + theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(                                              
        x="Features",
        y="Cluster",
        title="K-Means Clustering Baseball Prospects",
        subtitle="Blue = Below Avg, Red = Above Avg"
    )
    + coord_flip()                                       
)

g_heat_1

##how many in each cluster made majors in training data?
train_merged = pd.merge(career_train, career_identifiers, on='PlayerId')
promotion_count = train_merged.groupby('cluster')['Called_Up'].value_counts().reset_index(name='count')
total    = promotion_count.groupby('cluster')['count'].sum()
promoted = promotion_count[promotion_count['Called_Up'] == True].groupby('cluster')['count'].sum()
promotion_rate = promoted / total
##plot something including promotion rate




##Mlb readiness score based on cluster centroids 
##X = career_train.drop(columns=['PlayerId','cluster'])
##X_scaled = scaler.transform(X.values)

distances_to_centroids = kmeans.transform(X_train_scaled) 

readiness_score=[]
for player in range(len(career_train)):
    dists = distances_to_centroids[player]
    nearest = np.argmin(dists)
    base_score = promotion_rate[nearest]
    
    other_dists = np.delete(dists, nearest)
    margin = 1 - (dists[nearest] / other_dists.mean())
    margin = np.clip(margin, 0, 1)
    
    global_rate = promotion_rate.mean()
    score = global_rate + margin * (base_score - global_rate)
    readiness_score.append(score)

career_train['Readiness_Score'] = (pd.Series(readiness_score) * 100).round(1)


top_prospects=career_train.sort_values(by='Readiness_Score')
##print(career_train.groupby('cluster')[['Readiness_Score','Called_Up']].mean().round(2))
# top uncalled up prospects
##gems = career_train[career_train['Called_Up']==0].sort_values('Readiness_Score', ascending=False)


# test on test set
##X_test_scaled = scaler.transform(career_test.drop(columns='PlayerId',))
test_distances = kmeans.transform(X_test_scaled)
test_scores = []
global_rate = promotion_rate.mean()

for i in range(len(career_test)):
    dists = test_distances[i]
    nearest = np.argmin(dists)
    base_score = promotion_rate[nearest]
    
    other_dists = np.delete(dists, nearest)
    margin = 1 - (dists[nearest] / (other_dists.mean() + 1e-6)) 
    margin = np.clip(margin, 0, 1)
    
    score = global_rate + margin * (base_score - global_rate)
    test_scores.append(score)

career_test['Readiness_Score'] = np.array(test_scores) * 100

validation_df = pd.merge(career_test, career_identifiers[['PlayerId', 'Called_Up']], on='PlayerId')

validation_df.sort_values('Readiness_Score',ascending=False).head(25)

# Merge the training results with their names/status
train_results = pd.merge(career_train, career_identifiers, on='PlayerId', how='left')
test_results = pd.merge(career_test, career_identifiers, on='PlayerId', how='left')
final_df=pd.concat([train_results,test_results], axis=0)
final_df['Trajectory Score']=career['Trajectory_Score']

import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_prob = test_results['Readiness_Score'] / 100
y_true = test_results['Called_Up'].astype(int)
threshold = 0.60
y_pred = (y_prob >= threshold).astype(int)
accuracy = accuracy_score(y_true, y_pred)
cm= confusion_matrix(y_true,y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Called Up', 'Called Up'],
    yticklabels=['Not Called Up', 'Called Up']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold = {threshold})')
plt.tight_layout()
plt.show()


# Calculate AUC
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import matplotlib.pyplot as plt

y_true = validation_df['Called_Up'].astype(int)
y_prob = validation_df['Readiness_Score'] / 100
auc = roc_auc_score(y_true, y_prob)

# Plot Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: MLB Readiness Score')
plt.legend()
plt.show()


#create player comparison denograms

career_labels=career['PlayerId']
career_subset=career.drop('PlayerId',axis=1)
career_scaled=scaler.transform(career_subset.values)

dist_vec = pdist(career_scaled, metric='euclidean')
dist_mat = squareform(dist_vec)

hc = linkage(dist_vec, method='ward')

from scipy.spatial.distance import cdist

##Mike Trout Closest 25 Comps

target_id = 10155

target_idx = career[career['PlayerId'].astype(str) == str(target_id)].index[0]
target_vector = career_scaled[target_idx].reshape(1, -1)

dists = cdist(target_vector, career_scaled, metric='euclidean').flatten()
closest_indices = dists.argsort()[:25]


sub_scaled = career_scaled[closest_indices]
sub_labels = career_identifiers.iloc[closest_indices]['Name'].tolist()


plt.figure(figsize=(10, 8))
sub_hc = linkage(sub_scaled, method='ward')
dendrogram(sub_hc, labels=sub_labels, orientation='right')
plt.title("Closest Comparisons for Mike Trout")
plt.show()


##Elvis Andrus Closest 25 Comps

target_id_2 = 8709


target_idx = career[career['PlayerId'].astype(str) == str(target_id_2)].index[0]
target_vector = career_scaled[target_idx].reshape(1, -1)

# 2. Find the 20 most similar players (including established MLB guys)
dists = cdist(target_vector, career_scaled, metric='euclidean').flatten()
closest_indices = dists.argsort()[:25]


sub_scaled = career_scaled[closest_indices]
sub_labels = career_identifiers.iloc[closest_indices]['Name'].tolist()


plt.figure(figsize=(10, 8))
sub_hc = linkage(sub_scaled, method='ward')
dendrogram(sub_hc, labels=sub_labels, orientation='right')
plt.title("Closest Comparisons for Elvis Andrus")
plt.show()



#Iso Forrest Outlier detector
from sklearn.ensemble import IsolationForest

X_full = career.drop(columns=['PlayerId'])
full_scaler = StandardScaler()
X_full_scaled = full_scaler.fit_transform(X_full.values)

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)


final_df['Is_Outlier'] = iso.fit_predict(X_full_scaled)
raw_scores = iso.decision_function(X_full_scaled)

final_df['Outlier_Score'] = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

pca_full = PCA(n_components=2)
coords_full = pca_full.fit_transform(X_full_scaled)

viz_data = pd.DataFrame(coords_full, columns=['PC1', 'PC2'])
scores_series = final_df['Outlier_Score'].reset_index(drop=True)
viz_data['Outlier_Score'] = scores_series


plot=(
    ggplot(viz_data)
    + aes(x='PC1', y='PC2', color='Outlier_Score')
    + geom_point(alpha=0.6)
    + scale_color_gradient(low="blue", high="red")
     + theme(                                             
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        panel_background=element_blank()
    )
    + labs(title="Full Dataset: Isolation Forest Uniqueness Map")
)
plot

unicorn_subset=final_df[(final_df['Called_Up']==False) & (final_df['Is_Outlier']==-1)&(final_df['Last_Season']==2025)]
unicorn_subset.sort_values(by=['Outlier_Score','Readiness_Score'])
final_df

