# Dash app to plot embeddings

This is a dynamic dash app that can read in *precomputed* embeddings, perform dimension reduction and clustering.

Configuration (data file location, etc.), and parameter (arguments to dimension reduction and clustering algorithms) information is read from `ini` format files.  

The application can also update the algorithm parameters dynamically to change the conformation (shape) of the dimension reduction data. When the parameters are updated, a recomputation can be triggered. 

Important parameter information should be displayed so that it can be added to the plots (text box?), and exported as figures.
