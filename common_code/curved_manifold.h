
#ifndef curved_manifold_h_
#define curved_manifold_h_

#include <deal.II/grid/manifold.h>

#include <memory>


// small deformation in mesh to avoid triggering the constant Jacobian case
template <int dim>
class MyManifold : public dealii::ChartManifold<dim, dim, dim>
{
public:
  MyManifold()
    : factor(0.1)
  {}

  virtual std::unique_ptr<dealii::Manifold<dim>>
  clone() const override
  {
    return std::make_unique<MyManifold<dim>>();
  }

  virtual dealii::Point<dim>
  push_forward(const dealii::Point<dim> &p) const override
  {
    double sinval = factor;
    for (unsigned int d = 0; d < dim; ++d)
      sinval *= std::sin(dealii::numbers::PI * p[d]);
    dealii::Point<dim> out;
    for (unsigned int d = 0; d < dim; ++d)
      out[d] = p[d] + sinval;
    return out;
  }

  virtual dealii::Point<dim>
  pull_back(const dealii::Point<dim> &p) const override
  {
    dealii::Point<dim> x = p;
    dealii::Point<dim> one;
    for (unsigned int d = 0; d < dim; ++d)
      one(d) = 1.;

    // Newton iteration to solve the nonlinear equation given by the point
    dealii::Tensor<1, dim> sinvals;
    for (unsigned int d = 0; d < dim; ++d)
      sinvals[d] = std::sin(dealii::numbers::PI * x(d));

    double sinval = factor;
    for (unsigned int d = 0; d < dim; ++d)
      sinval *= sinvals[d];
    dealii::Tensor<1, dim> residual = p - x - sinval * one;
    unsigned int           its      = 0;
    while (residual.norm() > 1e-12 && its < 100)
      {
        dealii::Tensor<2, dim> jacobian;
        for (unsigned int d = 0; d < dim; ++d)
          jacobian[d][d] = 1.;
        for (unsigned int d = 0; d < dim; ++d)
          {
            double sinval_der = factor * dealii::numbers::PI * std::cos(dealii::numbers::PI * x(d));
            for (unsigned int e = 0; e < dim; ++e)
              if (e != d)
                sinval_der *= sinvals[e];
            for (unsigned int e = 0; e < dim; ++e)
              jacobian[e][d] += sinval_der;
          }

        x += dealii::invert(jacobian) * residual;

        for (unsigned int d = 0; d < dim; ++d)
          sinvals[d] = std::sin(dealii::numbers::PI * x(d));

        sinval = factor;
        for (unsigned int d = 0; d < dim; ++d)
          sinval *= sinvals[d];
        residual = p - x - sinval * one;
        ++its;
      }
    AssertThrow(residual.norm() < 1e-12, dealii::ExcMessage("Newton for point did not converge."));
    return x;
  }

private:
  const double factor;
};


void
warmup_code()
{
  // do some short burst of work to get all cores up by doing some useless
  // things; the final check is to prevent the compiler to optimize this away.
  {
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<double> big_vector(5000000);
    big_vector[10]                    = 10;
    big_vector[50]                    = 8;
    big_vector[big_vector.size() / 2] = 10;
    for (std::size_t j = 0; j < 50; ++j)
      for (std::size_t i = 0; i < big_vector.size(); ++i)
        big_vector[i] = big_vector[std::min(i + j, big_vector.size() - 1)] + big_vector[i];
    AssertThrow(big_vector[0] > 0, dealii::ExcMessage("Wrong computation"));
  }
}


#endif
