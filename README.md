### What is the Squealer?
The Squealer is an accessibility and educational tool designed to allow the user to adjust the fitted line of a simple 
(1D) linear regression problem or the separating line of a 2D logistic regression task. As the line gets moved around the space, the user 
receives visual and audio feedback regarding the quality of the new fit. In the case of linear regression,
the color of a point is its residual squared, while the frequency of the pitch corresponds to the residual sum of squares, 
with higher pitches being worse. For logistic regression, the shape of the point signifies its true classification and the color
is its distance from the line (more explicitly, the color is determined by the signed margin, so large positive values 
are good while negative values are a sign of misclassification). The tone played for logistic regression is a sum of 
squared margins of the misclassified data points (so a higher pitch is still worse). 

The graph at the bottom of the screen keeps track of the intercept and slope of the models tried so far. The color of the
points conveys the goodness of fit in the form of the log likelihood.

### How to Use the Squealer
At the top of the screen, you will see a graph of the data and its best linear model (as selected by internal fit() functions). 
On the line sit two large dots-- the handlebars. The right-most handlebar is controlled by the arrow keys,
while the left-most handlebar is controlled by W, A, S, and D. Moving these handlebars will in turn move the line that
connects them; it is this new proposed model that determines the residuals/ margins and pitch played. If the keys are
held down or are otherwise pressed "too close together," the program may hang, so please be careful! This issue can be
somewhat avoided by adjusting the step-size of the handlebars. The amount the
handlebars move on each click can be adjusted by the slider in the middle (but after you adjust the slider, click on the
instruction text box to "refocus" the code, or else the program will not know if you are trying to interact with the 
slider or graph). As the model line updates, so too will the coefficient tracking graph at the bottom of the screen. 

Currently, the mode of the model (whether we're performing a linear or logistic regression) is a global variable that
must be changed within the code; I hope to add a button so that the models can be switched within the dashboard.

### What's Inside?
This code uses ReactPy to make the user interface (UI) and Plotly (and json) to make the graphs; the starting/ best models 
are found with NumPy's polyfit function or sklearn's linear Logistic Regression package (depending on if you are 
performing linear or logistic regression, respectively). The data is generated using Numpy random functionss, but I hope,
in the future, the program can be modified to accept user-inputted data.

### Next Steps
Short term, I would like users to be able to switch between model types or even generate new data via buttons within the 
dashboard instead of having to modify the code internally or stop and rerun the program. There were also technical
difficulties surrounding auto play and html focus; Google would not allow tones to be played before a "user gesture on 
the page," even though the tone does not play until after a key is pressed to move a handlebar... Furthermore, I have not
figured out a way to ensure that pressing the keys will always move the handlebars, even after interacting with a different
part of the screen (but I feel like there should be some sort of focus override that would allow this). Initially, I had 
really hoped to use drag and drop instead of key presses, but that proved quite difficult with the necessary calculations
and Plotly redrawings; D3 has more dynamic user-interaction capabilities, so it might make sense to switch to D3 in the future. 

Aside from those UI issues, other next steps include allowing for user-inputted data, higher dimensional data (though that
begs the question, how can we visualize that space in a way that's intuitive for moving a model around?), or non-linear models. 




