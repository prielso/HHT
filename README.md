# HHT
Hyperbola Hough Transform.
#### (Run the main_example for a small presentation of the HHT algorithm)

### Introduction
The purpose of the Hough transform technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform
Simply put, Hough Transform looks for a specific shape (generally, that has a mathematical expression) in a boolean image. The basic case of Hough transform is detecting straight lines. the straight-line $y = mx + b$ can be represented as a point $(b, m)$ in the parameter space.
Say we have 2 points in the image ( $x_1$ , $y_1$ ),( $x_2$ , $y_2$ ) their presentation in the parameters space (b, m) will be: $b=-x_im+y_i$ which is the mathematical expression of a straight-line ( $-x_i$ is the slope and $y_i$ is the b-Axis intercept. So, a point in $(x, y)$ will transform to a line in the latent space $(b, m)$. practically, we will run through all m's and add 1 vote to the point $(m, -x_im+y_i)$.

![image](https://user-images.githubusercontent.com/93134859/200165660-b4a8540a-fe0e-4395-9221-78181590c1c3.png)

This feature extraction technique is applied for many analytical shapes, which in our algorithm is the Hyperbola with the mathematical expression:
$y=a_0+a_1 (x-a_2 )^2$. Where the parameters are:

$a_0$ - The y coordinate of the hyperbola peak.

$a_1$ – The convexity of the Hyperbola shape.

$a_2$ – The x coordinate of the Hyperbola peak.


The HHT algorithm input is a boolean image which basically shows the edges of the image like so:

![image](https://user-images.githubusercontent.com/93134859/200162986-967f101b-993d-4f7c-80e7-8fd13863975c.png)

Then by going through all the edges pixels $(x_i,y_i)$ we vote '1' for every possible combination of $a_0, a_1, a_2$ that agrees with the equation:
![image](https://user-images.githubusercontent.com/93134859/200163040-0ec634fe-b8ec-4d07-ac33-a2be19f9a16c.png)

for every known edge pixel coordinate we run through all the feasible combinations and add '1' vote to that specific combination.
In the end we present the combination that got the most votes as the most likely Hyperbola in the image. 
We have developed few versions for the algorithm which differ mainly in the order which the Hyperbola's parameters are been "scanned".
<br>
#### This is a brute force solution for the HHT algorithm in pseudo-code:

HHT (image):

-Canny_image = canny_function(image)

-Parameters_space = array $(a2_{size}, a1_{size}, a0_{size})$

-For (x,y) ∈ Canny_image:

--For a0 ∈ (0,y):

---For a1 ∈ $( a1_{min},a1_{max} )$

--- $a2=x±\sqrt{((y-a0)/a1)}$

---Parameters_space [a2, a1, a0] += 1

return a2*,a1*,a0* = $argmax_{a2, a1, a0}$ ⁡Parameters_space $[a2,a1,a0]$

<br>In simple words, the algorithm takes an edge pixel and for every possible 
combination of a0 and a1 chooses a2 according to - $a2=x±\sqrt{((y-a0)/a1)}$
and adds one vote for that specific combination: $(a0,a1,x±\sqrt{((y-a0)/a1))}$
after finishing running over all the edges and voting accordingly we return the set of 
parameters that received the highest number of votes.
<br>
<br>
### Running the example on the given "example_2d_array.npy" will give you results like so:
![image](https://user-images.githubusercontent.com/93134859/200530822-d36dfb32-14e3-4b37-b473-f537ad713882.png)

