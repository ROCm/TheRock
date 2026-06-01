.. meta::
  :description: Composable Kernel mathematical basis
  :keywords: composable kernel, CK, ROCm, API, mathematics, algorithm

.. _supported-primitives:

********************************************************************
Composable Kernel mathematical basis
********************************************************************

This is an introduction to the math which underpins the algorithms implemented in Composable Kernel.


For vectors :math:`x^{(1)}, x^{(2)}, \ldots, x^{(T)}` of size :math:`B` you can decompose the
softmax of concatenated :math:`x = [ x^{(1)}\ | \ \ldots \ | \ x^{(T)} ]` as,

.. math::
   :nowrap:

   \begin{align}
      m(x) & = m( [ x^{(1)}\ | \ \ldots \ | \ x^{(T)} ] ) = \max( m(x^{(1)}),\ldots, m(x^{(T)}) )  \\
      f(x) & = [\exp( m(x^{(1)}) - m(x) ) f( x^{(1)} )\ | \ \ldots \ | \ \exp( m(x^{(T)}) - m(x) ) f( x^{(T)} )] \\
      z(x) & = \exp( m(x^{(1)}) - m(x) )\ z(x^{(1)}) + \ldots + \exp( m(x^{(T)}) - m(x) )\ z(x^{(1)}) \\
      \operatorname{softmax}(x) &= f(x)\ / \ z(x)
   \end{align}

where :math:`f(x^{(j)}) = \exp( x^{(j)} - m(x^{(j)}) )` is of size :math:`B` and
:math:`z(x^{(j)}) = f(x_1^{(j)})+ \ldots+ f(x_B^{(j)})` is a scalar.

For a matrix :math:`X` composed of :math:`T_r \times T_c` tiles, :math:`X_{ij}`, of size
:math:`B_r \times B_c` you can compute the row-wise softmax as follows.

For :math:`j` from :math:`1` to :math:`T_c`, and :math:`i` from :math:`1` to :math:`T_r` calculate,

.. math::
   :nowrap:

   \begin{align}
      \tilde{m}_{ij}   &= \operatorname{rowmax}( X_{ij} ) \\
      \tilde{P}_{ij}   &= \exp(X_{ij} - \tilde{m}_{ij} ) \\
      \tilde{z}_{ij}   &= \operatorname{rowsum}( P_{ij} ) \\
   \end{align}

If :math:`j=1`, initialize running max, running sum, and the first column block of the output,

.. math::
   :nowrap:

   \begin{align}
      m_i            &= \tilde{m}_{i1} \\
      z_i            &= \tilde{z}_{i1} \\
      \tilde{Y}_{i1} &= \diag(\tilde{z}_{ij})^{-1} \tilde{P}_{i1}
   \end{align}

Else if :math:`j>1`,

1. Update running max, running sum and column blocks :math:`k=1` to :math:`k=j-1`

.. math::
   :nowrap:

   \begin{align}
      m^{new}_i &= \max(m_i, \tilde{m}_{ij} ) \\
      z^{new}_i &= \exp(m_i - m^{new}_i)\ z_i + \exp( \tilde{m}_{ij} - m^{new}_i )\ \tilde{z}_{ij}  \\
      Y_{ik}    &= \diag(z^{new}_{i})^{-1} \diag(z_{i}) \exp(m_i - m^{new}_i)\ Y_{ik}
   \end{align}

2. Initialize column block :math:`j` of output and reset running max and running sum variables:

.. math::
   :nowrap:

   \begin{align}
      \tilde{Y}_{ij} &= \diag(z^{new}_{i})^{-1} \exp(\tilde{m}_{ij} - m^{new}_i ) \tilde{P}_{ij} \\
      z_i            &= z^{new}_i \\
      m_i            &= m^{new}_i \\
   \end{align}
