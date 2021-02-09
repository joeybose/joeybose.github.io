---
layout: post
title:  "Adversarial Attacks Part1!"
date:   2018-03-29 10:08:27 -0500
permalink: /Blog/AdvAttacks
comments: False
share: True
categories: Adversarial Attacks
---

I've been looking at Adversarial Attacks a lot recently and I figured I'd make
a series of posts about them. Frankly, the number of papers being published on
advesarial attacks is a bit ridiculous and no one has the time to read all of
them. As a result, I want to create this series especially for people who are
getting excited about this space but reading 10 papers about essentially the
same idea but perturbed by a small amount is not practical. Figuring out what is actually
a good idea versus a cute hack is difficult but I hope to capture what I feel
are the big interesting ideas in this field. So without further ado lets start
at one of the first big attacks: Fast Gradient Sign Method.

# What are Adversarial Attacks?
Adversarial attacks are perturbations that are often small but generally
imperceptible to humans with the goal of deteriorating the performance of
Machine Learning models. In practice, this could be adding small but
specifically crafted noise to an image so that a trained Convolutional Neural
Net misclassifies the central object in the image. If pictures are worth a
thousand words then examples are worth a million right? Here's an illustrative example taken
from the FGSM paper.

<img src="../assets/my_assets/adv_attacks/adv_example1.png" alt="" style="border-radius:0%;height: 50%;width: 90%;margin-left: auto;margin-right: auto;position: relative;display: block; padding-top: 30px; padding-bottom: 30px">
<!--![EX1]({{ "../assets/adv_attacks/adv_example1.png" | absolute_url }}){: style="display: block; margin: auto;"}-->

By adding a small amount of noise to the original image the attack was able to
change the classification of the original image of a panda with high confidence
to that of a gibbon with also high confidence. Also this would all be for
naught if the attacked image didn't actually look like the original image to
humans!

# Why should we care about Adversarial Attacks?
Machine Learning models are doing amazing things in many different industries,
from accelerating the self-driving car industry, to predictive analytics that drive
large business decisions, to agents that learn to play games at super human levels.
As active members of this community we have a lot of power in shaping the future
of this field, and with great power comes even greater responsibility. Now consider a
scenario where a malicious attacker is trying to attack
critical systems that rely on these algorithms. For example, imagine attacking
the computer vision system of a self-driving car to prevent it from detecting
stop signs, clearly the potential for damage can be enormous. From a research
perspective it's important to know the failure modes of a particular algorithm
so that we can work towards making them better and more robust. Ok enough talk
lets dive into the details about the first algorithm.

# Intuition for Fast Gradient Sign Attack
It is natural to think that if you have a trained classification model that it
should assign the same class to similar inputs. In the case of images, each
pixel is encoded with 8-bit precision (0-255 RGB values) so typically we expect
that if you have an image $$x$$ and a perturbed version of the image $$\tilde{x} = x + \eta$$
to have the same class assigned by the model if the perturbation is small enough.
Small enough here can be a bit vague but one way to think about it is as
follows. If we're working with only 8-bit precision that means perturbations
smaller than $$1/255$$ per pixel would be discarded in terms of the data
encoded. Now what happens when we do a dot product with the perturbed image?


$$
\begin{align*}
    w^T\tilde{x} = w^Tx + w^T\eta
\end{align*}
$$



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
