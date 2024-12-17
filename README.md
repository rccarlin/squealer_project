### Summary
The Squealer is an accessibility and educational tool designed to allow the user to adjust the fitted line of a simple (1D) linear regression problem or (hopefully soon) a 2D logistic regression task. As the line gets moved around the space, the user receives visual and audio feedback regarding the quality of the new fit. In the case of the linear regression, the color of the points and the frequency of the pitch played correspond to the residual sum of squares (RSS). For logistic regression, RSS will not be used, but will instead be replaced by either loss or somthing that considers the margin (or possibly both, I have not gotten that far yet).

### How to Use the Squealer
At the top of the screen, you will see a graph with the data and the line fit using Numpy's polyfit function. On the line sit two larger dots-- the handlebars. Moving these handlebars will move the fitted line they are attached to. The handlebars can be moved up, down, left, and right, and at a step-size determined byt the slider at the bottom of the page. (If you do adjust the step size, you may need to "refocus" the program by clicking on the instructions text box so that the program knows you want to interact with teh graph and not the slider anymore.)
For now, the left-most handlebar is controlled by the W, A, S, and D keys, while the right-most is controlled by the arrow keys.

