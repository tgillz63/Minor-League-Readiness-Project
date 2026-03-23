## Determing Major League Readiness in Baseball Prospects With Machine Learning 

#Overview 
The timing of a Major League Baseball star’s callup is a very delicate decision to handle. Call them up too early and struggles can ruin their confidence. Letting them rot away too long in the minors can cause a rift. The player may think management either doesn’t believe in them or is manipulating their service time to save money. But what if there was a way to dtermine if the player was ready, beyond just the eye test or gut feel?

My project aims to quanify major league readiness using K-Means clustering, K-Nearest neighbors hierarchical clustering denograms, and identifying outlier prospects with an isolation forrest. This project implements a multi-stage machine learning pipeline to evaluate baseball prospects. It goes beyond raw counting stats by adjusting for age/level, calculating development trajectories, and using unsupervised learning to identify "MLB Readiness" scores.

#Key Features 
1. **Age Adjusted Performance**- wOBA, ISO, and wRC+ normalized based on difference between player's age and the minor league leve;'s average age for that particular year.
2. **Trajectory Score**- Calculating the overall rate of improvement or decline in plate discipline and power metrics over a player's minor league career thus far.
3. **K-Means Clustering**- K-Means clustering is used to seperate players into one of 5 player archetypes. The cluster promotion rate and each player's distance from the cluster centroid is used to create an individual MLB readiness score.
<img width="600" height="193" alt="Screenshot 2026-03-23 at 5 01 21 PM" src="https://github.com/user-attachments/assets/efe82efb-49b3-4526-8fce-e5d28f20505c" />

5. **K-Nearest Neighbors Heirarchial Clustering** - Used to identify closest comparisons for MLB stars such as Mike Trout or Elvis Andrus. Can be adjusted to any player/ number of closest neigbors by inputting the players fangraphs id for targetid, and adjusting the slice indexing in closest_indexes.
```
##Mike Trout Closest 25 Comps

target_id = 10155

target_idx = career[career['PlayerId'].astype(str) == str(target_id)].index[0]
target_vector = career_scaled[target_idx].reshape(1, -1)

dists = cdist(target_vector, career_scaled, metric='euclidean').flatten()
closest_indices = dists.argsort()[:25]
```

<img width="600" height="236" alt="Screenshot 2026-03-23 at 5 02 38 PM" src="https://github.com/user-attachments/assets/fd66f7a8-76be-431f-b774-ad297f519e82" />

5. **"Unicorn" Prospect Finder** - Uses an isolation Forest to find statistical outliers by assigning each prospect an outlier score relative to the rest of the prospect pool.
<img width="319" height="243" alt="Screenshot 2026-03-23 at 5 03 53 PM" src="https://github.com/user-attachments/assets/45492d11-ba25-4c61-951b-5ad0038b1b26" />

#Necessary Installs
-scikit-learn plotnine matplotlib seaborn scipy

#Takeaways
Recently aquired Brewers shortstop Jett Williams appeared both as Mike Trout's closest comparison and as one of 5 "Unicorn" outlier prospecHad he still been with the Mets I would've reccomended agressively targetig him in a trade just as the Brewers did. The three unicorn prospects that happened to be outside MLB's top 100 prospects were Pablo Aliendo, Cade Doughty, and Ethan Workinger. I would reccomend these three players as low risk, low, cost but high reward underated trade targets given their outlier tools and lack of hype. 

```

