---
layout: post
title:  "Hyperbolic Normalizing Flows!"
date:   2020-06-16 10:08:27 -0500
comments: True
share: True
categories: Normalizing Flows
---
Normalizing Flows have been all the rage lately and with all new breeds of
generative models come new avenues for their application. Recently, there's
been a burgeoning interest in bringing in tools from differential geometry in
order to do effective Deep Learning on non-Euclidean manifolds. For the purposes of
this blog post I'll focus on one recent extension of Normalizing Flows to
hyperbolic spaces based on my recent ICML 2020 paper titled "Latent Variable
Modeling with Hyperbolic Normalizing Flows".
[arXiv](https://arxiv.org/pdf/2002.06336.pdf)
While a deep knowledge of Riemannian geometry is not needed it definitely helps
in understandings a lot of the intution and technical details. In this blog
post I'll try to distill the key concepts but to get the most out of the material
I higly recommend the following blog posts as well to get a feel for Hyperbolic
Geometry:

* [Source1](http://bjlkeng.github.io/posts/hyperbolic-geometry-and-poincare-embeddings/)
* [Source2](https://dawn.cs.stanford.edu/2018/03/19/hyperbolics/)
* [Source3](https://wiseodd.github.io/techblog/2019/02/22/riemannian-geometry/)

# Motivation
To motivate why we might want to care about geometry when doing generative
modelling we can simply look at different domains that have seen remarkable
breakthroughs.

![EX1]({{ "../assets/HyperFlow_15min/HyperFlow_15min.002.jpeg" | absolute_url }}){: style="display: block; margin: auto;"}

But in all these different domains, you can ask the following question: "What
do we know about the data already?". For instances, images are known to be very
grid-like, text is very discrete and structured (and arbitrary assortment of
words don't usually make a sentence) and even molecules can be represented as
graphs. Data in these domains have vastly different geometry, so the natural question is
shouldn't any generative model being trained also be privy this geometry?
Sometimes as practictioners we can neglect what is already available to us, and
neglecting known geometry in the data unnecessarily makes the learning problem
significantly harder.

# Hierarchical structure and Hyperbolic Geometry
Turning to a concrete use case, what if we know apriori that our data is
tree-like or has rich hierarchical structure? What is the right geometry here?
Well to gain intuition we can first see what can go wrong when we naively try
to embed a tree in Euclidean space.

![EX1]({{ "../assets/HyperFlow_15min/HyperFlow_15min.004.jpeg" | absolute_url }}){: style="display: block; margin: auto;"}
In this construction we evenly place all leaves around the circumference of an
imaginary circle with radius $$r$$.
Observe how the Euclidean distance between the pink and green nodes decreases
as we increase the depth of the tree but the graph distance ---i.e. shortest
path between these nodes, actuall increases! Clearly, Euclidean space is not
respecting this notion of graph distance well. This happens to be a fundamental
limitation of Euclidean spaces, and the hand wavy answer to this phenomenon is due
to the fact the space is not growing fast enough to
accomodate the exponential growth of nodes.
I love this figure (taken from [fig](https://openreview.net/pdf?id=BJg73xHtvr))
because it succinctly encapsulates the problem in embedding hierarchies in
Euclidean.

So lets see how hyperbolic spaces alleviates this problem:
![EX1]({{ "../assets/HyperFlow_15min/HyperFlow_15min.005.jpeg" | absolute_url }}){: style="display: block; margin: auto;"}
One important fact to realize is that because hyperbolic spaces are manifolds
of constant negative curvature many of the geometric intuitions from Euclidean
spaces go out the window. For instance the shortest path between two points is
now a curved path (geodesic)! So it just so happens the shortest path between
two nodes in a tree must go through a common parent, in this case this is the
root node, which matches our intuition of graph distances.

# Latent Variable Modeling and Normalizing Flows
Now consider the problem of latent variable modeling in the canonical VAE
setting. To train these models we often optimize for the Evidence Lower Bound (ELBO)
objective:

$$
\begin{align*}
\mathbb{E}_{q_i(z)}\Big[\log \frac{p(x|z)p(z)}{q_i(z)}\Big] = \mathbb{E}_{q_i(z)}[\log p(x_i|z)] - D_{KL}(q_i || p)
\end{align*}
$$
But what have we already implictly assumed in this formulation? All densities
are taken to be Euclidean densities even though the data can be highly
non-Euclidean! Even learning more flexible approxite posteriors through
Normalizing Flows cannot always overcome this fundamental geometric limitation.
What we really need is to learn densities on manifolds and in the particular
case of this blog post hyperbolic manifolds.

While there is an abundance of material on understanding normalizing flows they
can really be understood in terms of 3 key desiderata:

* Each function $$f_i$$￼ must be invertible.
* We must be able to efficiently sample from the final distribution $$z_j = f_j
  \circ f_{j-1} \circ \dots \circ f_1(z_0)$$
* We must be able to efficiently compute the associated change in volume


$$
\begin{align*}
\log p(z_j) = \log p(z_0) - \sum_{i=1}^k\log det\Big|\frac{\partial f_j}{\partial z_{j-1}} \Big|
\end{align*}
$$
Thus normalizing flows compose several invertible simple functions to construct
a arbitrarily complex function which allows one to sample from a simple base
distribution but yields a sample from a significantly more complex
distribution. All of this is hinged on the change in volume formula for
probability distributions which can be extermely efficient to compute given
certain types of $$f_i$$'s.

# Quick Primer on Lorentz Model of hyperbolic geometry
Hyperbolic geometry initself is a vast and rich topic that avid scholars can
pour years of their life into. For the purposes of this blog post I'll outline
the spark notes version and refer the interested reader to the Appendix of the
paper.

There are many equivalent models of hyperbolic geometry, but the Lorentz model
offers the simplest explicit formulas. At its core the a hyperboloid manifold
is equipped with the Lorentz Inner Product:

$$
\begin{align*}
\langle \textbf{x}, \textbf{y} \rangle_{\mathcal{L}} := -x_0y_0 + x_1y_1 + \dots + x_ny_n,
\end{align*}
$$

With this inner product (sometimes called metric) we can now define familiar notions of
distances and angles in hyperbolic space. Loosely speaking when a smooth
manifold equipped this metric is also a Riemannian manifold. Formally, the
hyperboloid with curvature constant $$K$$ is defined as:

$$
\begin{align*}
\mathbb{H}^{n}_K := \{x \in \mathbb{R}^{n+1}:  \langle \textbf{x}, \textbf{x} \rangle_{\mathcal{L}} = 1/K, \ x_0 > 0, \ K<0 \}
\end{align*}
$$

## Tangent Spaces and Parallel Transport
In laymans terms the tangent space at a point $$\mu$$ is a Euclidean space
spanned by all tangent vectors to the manifold at $$\mu$$. In pictures this is:

![EX1]({{ "../assets/HyperFlow_15min/HyperFlow_15min.009.jpeg" | absolute_url }}){: style="display: block; margin: auto;"}

We may also move vectors between to tangent spaces. The operation that does
this but "preserves" the metric is known as parallel transport. We'll see later
on that parallel transport doesn't induce any change in volume.

## Exponential and Logarithmic Maps
We can also move vectors from the tangent space at a point to the manifold and
vice-versa. Mapping a point from the tangent space to the manifold is known as
the exponential map, while the inverse operation is called the logarithmic map.
It's important to note that both maps may not have closed form solutions for
arbitrary Riemannian manifolds, but luckily do have nice closed forms in the
Lorentz model

## Distributions on Hyperbolic Spaces





The perturbation causes the pre-activation to increase by $$w^T\eta$$. We can
maximize this increase subject to a maxnorm constraint if we let $$\eta =
\textrm{sign}(w) $$. Further, if $$w$$ is $$n$$ dimensional and the average magnitude of
a vector in $$w$$ is $$m$$ then the total increase caused by the perturbation
is $$\epsilon m n$$. If we force $$\eta$$ to say an infinity norm constraint then it doesn't
grow with the dimensionality of the weight matrix! Ok that might not have been
so obvious but its clearer if you write out the definition of the infinity
norm. In words the infinity norm for a vector is the element which has the
maximum absolute value, and as result its independent from the vector length. We can now let small refer
to a value $$\epsilon$$ that would be discarded due to precision. With this in mind we can now
let $$\Vert  \eta \Vert_{\infty} < \epsilon $$. Ok but you may ask why this
is reasonable for Neural Nets which are much deeper and highly non-linear? Well
there are a few assumptions that make the intuition a bit trickier to extend to
the general Neural Net case. But imaging all our non-linearities are ReLu's,
which are picewise linear and perhaps the most popular activation function
then we're basically cutting up the input space linearly and then this is not
such a crazy idea.

# Fast Gradient Sign Attack
So the next logical question becomes how do we leverage our intuition to create
an attack? Actually its not so hard, if you take the most typical scenario
where you have a model parametrized by weights $$ \theta $$ and the goal is to
minimize a cost function $$ J(\theta,x,y) $$. For classification this would be
cross-entropy and $$x$$ are the inputs while $$y$$ are the labels. To construct
an attack we have to go in the opposite direction of minimizing the cost. That
is to say we want the weights to respond in a way that moves us in opposite
direction to the classification boundary until we hit another class and the
perturbed input is misclassified. To do this we can take a first order
approximation of the cost function with the current weights and with respect to
a particular input $$x$$. This is perhaps the easiest thing we can do as this
just amounts to taking the gradient of the cost function but with respect to
the inputs. So our perturbation then becomes:


$$
\begin{align*}
    \eta = \epsilon \textrm{sign}(\nabla_x J(\theta,x,y))
\end{align*}
$$

Wow that was super easy as we can leverage modern libraries like tensorflow and
pytorch to compute these perturbations easily. In pytorch code this looks like:

~~~ python
def attack(self, inputs, labels, model, *args):
    """
    Given a set of inputs and epsilon, return the perturbed inputs (as Variable objects),
    the predictions for the inputs from the model, and the percentage of inputs
    unsucessfully perturbed (i.e., model accuracy).
    The adversarial inputs is a python list of tensors.
    The predictions is a numpy array of classes, with length equal to the number of inputs.
    """
    adv_inputs = inputs.data + self.epsilon * torch.sign(inputs.grad.data)
    adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
    adv_inputs = Variable(adv_inputs, requires_grad=False)

    predictions = torch.max(model(adv_inputs).data, 1)[1].cpu().numpy()
    num_unperturbed = (predictions == labels.data.cpu().numpy()).sum()
    adv_inputs = [ adv_inputs[i] for i in range(inputs.size(0)) ]

    return adv_inputs, predictions, num_unperturbed
~~~
So how good is this attack? Well it's not too shabby as with an $$\epsilon=0.25$$ the attack can
break a softmax classifier with an error rate of $$99.9\%$$. However, the main selling point is how easy it is to craft
an adversarial sample.

# Conclusion
There are many other cool things to consider like the generalization of this attack and possible defences but this post is getting rather long so I’ll conclude with a small remark. The FGSM attack is not meant to be a strong attack but rather a fast one. There are other stronger attacks which are much harder to defend against such as Carlini-Wagner but they take much longer to construct. But i’ll shelve that discussion for a future post. As an overall note, I’m hoping to dedicate each post to a specific attack or defense and really look at it in detail. In the future I hope to talk more about some of the research that I’m doing or research directions in that I’m most excited about. Feel free to reach out with questions, comments and especially mistakes I make along the way!


{% if page.comments %}

<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://joeybose.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

{% endif %}
