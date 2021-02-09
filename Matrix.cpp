// Matrix Class
// Ben Crabtree, 2021

# include "Matrix.hpp"
# include <vector>
# include <utility> // std::pair,
# include <string>
# include <iostream>
# include <cmath>

Matrix::Matrix()
{
    Rows = 0;
    Cols = 0;
    Augmented = false;
    REF = false;
    Vals = {{}};
}

Matrix::Matrix(int rows, int cols, int matrixType)
{
    Rows = rows;
    Cols = cols;
    Augmented = false;
    REF = false;
    //Transposed = false;
    
    // matrixType == 0 gives Zero matrix
    if (matrixType == 0)
    {
        std::vector<std::vector<float>> vals(Rows, std::vector<float>(Cols, 0));
        Vals = vals;
    }
    // matrixType == 1 gives matrix of ones
    else if (matrixType == 1)
    {
        std::vector<std::vector<float>> vals(Rows, std::vector<float>(Cols, 1));
        Vals = vals;
    }
    // matrixType == 2 gives identity matrix
    else if (matrixType == 2)
    {
        std::vector<std::vector<float>> vals;
        if (rows == cols)
        {
            for (int i = 0; i < rows; i++)
            {
                std::vector<float> row;
                for (int j = 0; j < cols; j++)
                {
                    if (i == j)
                    {
                        row.push_back(1);
                    }
                    else
                    {
                        row.push_back(0);
                    }
                }
                vals.push_back(row);
            }
            Vals = vals;
        }
        else
        {
            std::cerr << "Error: Matrix must be square." << std::endl;
            throw "Matrix must be square error.";
        }
    }
    else
    {
        std::cerr << "Error: Please enter a matrix type or values for matrix." << std::endl;
        throw "No matrix values entered error.";
    }
}

// General Constructor
Matrix::Matrix(int rows, int cols, std::vector<std::vector<float>> vals)
{
    Rows = rows;
    Cols = cols;
    Augmented = false;
    REF = false;
    //Transposed = false;

    // Check number of rows in vals matches rows
    if ((int)vals.size() != rows)
    {
        std::cerr << "Error: Dimensions of input vals must match number of rows and cols given" << std::endl;
        throw ("Num values must match");
    }
    // Check number of columns in vals matches cols
    for (int i = 0; i < rows; i++)
    {
        if ((int)vals[i].size() != cols)
        {
            std::cerr << "Error: Dimensions of input vals must match number of rows and cols given." << std::endl;
            throw ("Input dimension error.");
        }
    }
    std::cout << "Num vals match dimensions: (" << rows << ", " << cols << ")" << std::endl;
    Vals = vals;
    std::cout << "Constructed Matrix." << std::endl;
}

Matrix::~Matrix()
{
    // Destructor
}

int Matrix::get_Rows()
{
    return Rows;
}

int Matrix::get_Cols()
{
    return Cols;
}

std::vector<std::vector<float>> Matrix::get_Vals()
{
    return Vals;
}

void Matrix::printMatrix()
{
    if (Rows == 0 && Cols == 0)
    {
        std::cout << "Matrix is Empty." << std::endl;
        std::cout << "[]" << std::endl;
    }
    else
    {
        if (!Augmented)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    std::cout << Vals[i][j] << " ";
                }
                std::cout << " " << std::endl;
            }
        }
        else
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    std::cout << Vals[i][j] << " ";
                    if (j == Cols - 2)
                    {
                        std::cout << "| ";
                    }
                }
                std::cout << " " << std::endl;
            }
        }
    }
}

void Matrix::transpose()
{
    // If square matrix, swap elements in place
    if (Rows == Cols)
    {
        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Cols; j++)
            {
                float temp;
                if (i < j && Vals[i][j] != Vals[j][i])
                {
                    temp = Vals[i][j];
                    Vals[i][j] = Vals[j][i];
                    Vals[j][i] = temp;
                }
            }
        }
    }
    // Matrix not square, create new 2D Vals vector
    else
    {
        std::vector<std::vector<float>> newVals;
        for (int j = 0; j < Cols; j++)
        {
            std::vector<float> row;
            for (int i = 0; i < Rows; i++)
            {
                row.push_back(Vals[i][j]);
            }
            newVals.push_back(row);
        }
        Vals = newVals;
    }
    float temp;
    temp = Rows;
    Rows = Cols;
    Cols = temp;
    REF = false;
}

void Matrix::printDims()
{
    std::cout << "(" << Rows << ", " << Cols << ")" << std::endl;
}

void Matrix::setAttributes(int rows, int cols, std::vector<std::vector<float>> vals)
{
    Rows = rows;
    Cols = cols;
        // Check number of rows in vals matches rows
    if ((int)vals.size() != rows)
    {
        std::cerr << "Error: Dimensions of input vals must match number of rows and cols given" << std::endl;
        throw ("Num values must match");
    }
    // Check number of columns in vals matches cols
    for (int i = 0; i < rows; i++)
    {
        if ((int)vals[i].size() != cols)
        {
            std::cerr << "Error: Dimensions of input vals must match number of rows and cols given." << std::endl;
            throw ("Input dimension error.");
        }
    }
    std::cout << "Num vals match dimensions: (" << rows << ", " << cols << ")" << std::endl;
    Vals = vals;
    std::cout << "Set Rows, Columns and Vals." << std::endl;
}

// Returns the l_p,q matrix norm.
// Eg set p = 2, q = 1 for the l_2,1 norm
// or set p = 2, q = 2 for the Frobenius norm.
float Matrix::lpqNorm(float p, float q)
{
    float outerSum = 0;
    for (int j = 0; j < Cols; j++)
    {
        float innerSum = 0;
        for (int i = 0; i < Rows; i++)
        {
            innerSum += std::pow(Vals[i][j], (int)p);
        }

        float innerRoot = std::pow(innerSum, q/p);
        outerSum += innerRoot;
    }
    float outerRoot = std::pow(outerSum, 1.0/q);
    return outerRoot;
}

Matrix operator + (Matrix const & M, Matrix const & N)
{
    Matrix result = Matrix(M.Rows, M.Cols, 0);
    if (M.Cols == N.Cols && M.Rows == N.Rows)
    {
        for (int i = 0; i < M.Rows; i++)
        {
            for (int j = 0; j < M.Cols; j++)
            {
                result.Vals[i][j] = M.Vals[i][j] + N.Vals[i][j];
            }
        }
        return result;
    }
    else
    {
        std::cerr << "Error: Matrix dimensions must match to add.";
        throw "Matrix dimension mismatch error.";
    }
}

// Scalar Multiplication
Matrix operator * (float const & a, Matrix const & M)
{
    Matrix result = Matrix(M.Rows, M.Cols, 0);
    for (int i = 0; i < M.Rows; i++)
    {
        for (int j = 0; j < M.Cols; j++)
        {
            result.Vals[i][j] = M.Vals[i][j] * a;
        }
    }
    return result;
}

// Matrix Multiplication - Naive implementation
// Runs in O(N^3) when multiplying two NxN matrices
// Runs in O(LMN) when multiplying LxM and MxN matrices
Matrix operator * (Matrix const & M, Matrix const & N)
{
    Matrix result = Matrix(M.Rows, N.Cols, 0);
    if (M.Cols == N.Rows)
    {
        for (int i = 0; i < M.Rows; i++)
        {
            for (int j = 0; j < N.Cols; j++)
            {
                for (int k = 0; k < M.Cols; k++)
                {
                    result.Vals[i][j] += M.Vals[i][k] * N.Vals[k][j];
                }
            }
        }
        return result;
    }
    else
    {
        std::cerr << "Error: Matrix inner dimensions must match to multiply.";
        throw "Matrix dimension mismatch error.";
    }
}

Matrix operator - (Matrix const & M, Matrix const & N)
{
    Matrix result = Matrix(M.Rows, M.Cols, 0);
    if (M.Cols == N.Cols && M.Rows == N.Rows)
    {
        for (int i = 0; i < M.Rows; i++)
        {
            for (int j = 0; j < M.Cols; j++)
            {
                result.Vals[i][j] = M.Vals[i][j] - N.Vals[i][j];
            }
        }
        return result;
    }
    else
    {
        std::cerr << "Error: Matrix dimensions must match to subtract.";
        throw "Matrix dimension mismatch error.";
    }
}

// Row Operations

// Adds row2 to row1 (row1 altered)
void Matrix::rowAdd(int row1, int row2)
{
    for (int i = 0; i < Cols; i++)
    {
        Vals[row1][i] += Vals[row2][i];
    }
}

void Matrix::rowScale(int row, float c)
{
    if (c != 0)
    {
        for (int i = 0; i < Cols; i++)
        {
            Vals[row][i] *= c;
        }
    }
    else
    {
        std::cerr << "Error: c must be a non zero constant." << std::endl;
        throw "Can't scale row by 0 error.";
    }
}

void Matrix::rowSwap(int row1, int row2)
{
    std::vector<float> temp;
    for (int i = 0; i < Cols; i++)
    {
        temp.push_back(Vals[row1][i]);
        Vals[row1][i] = Vals[row2][i];
        Vals[row2][i] = temp[i];
    }
}

void Matrix::augment(std::vector<float> b)
{
    if (Rows == (int)b.size())
    {
        for (int i = 0; i < Rows; i++)
        {
            Vals[i].push_back(b[i]);
        }
        std::cout << "Augmented matrix." << std::endl;
        Cols += 1;
        Augmented = true;
    }
    else
    {
        std::cerr << "Error: Vector b must be same length as number of Rows.";
        throw "Length mismatch error.";
    }
}

void Matrix::deaugment()
{
    for (int i = 0; i < Rows; i++)
    {
            Vals[i].pop_back();
    }
        std::cout << "Deaugmented matrix." << std::endl;
        Cols -= 1;
        Augmented = false;
}

// Gaussian elimination to put matrix in row eschelon form
// From https://en.wikipedia.org/wiki/Row_echelon_form, REF has the following conditions:
// 1. All rows consisting of only zeroes are at the bottom.
// 2. The leading coefficient (also called the pivot) of a nonzero row is always strictly to the right
//    of the leading coefficient of the row above it.
// Note: leading pivot element not necessarily 1 in this implementation.
// Returns a float scaling_factor for use in calculating the determinant after row reducing.
float Matrix::gaussElim()
{
    if (Augmented == true && Rows < Cols-1)
    {
        std::cout << "Matrix is underdetermined." << std::endl;
    }
    else if (Augmented == true && Cols-1 < Rows)
    {
        std::cout << "Matrix is overdetermined." << std::endl;
    }
    float scaling_factor = 1;
    if (Rows > 1)
    {
        // For each column (bounded by number of rows)
        for (int i = 0; i < Rows; i++)
        {
            // Find max element in column and the row it is in
            float maxEl = std::abs(Vals[i][i]);
            int maxRow = i;
            for (int j = i + 1; j < Rows; j++)
            {
                if (std::abs(Vals[j][i]) > maxEl)
                {
                    maxEl = Vals[j][i];
                    maxRow = j;
                }
            }
            // Swap row with max elem with current row
            if (i != maxRow)
            {
                rowSwap(i, maxRow);
                scaling_factor *= -1;
                printMatrix();
            }            
            // Make all elements in column below max elem (pivot) 0 using row operations
            std::cout << std::endl;
            for (int j = i+1; j < Rows; j++)
            {
                if (Vals[j][i] != 0)
                {
                    float c = -maxEl / Vals[j][i];
                    rowScale(j, c);
                    scaling_factor /= c;
                    rowAdd(j, i);
                    printMatrix();
                }
                std::cout << std::endl;
            }
        }
        REF = true;
    }
    return scaling_factor;
}

// Solve system of linear equations
// Takes a matrix, augments it with column vector b, puts it in REF
// and solves A*x = b for vector x using back substitution.
// Returns a column vector x in the form of a Matrix object.
Matrix Matrix::solve(std::vector<float> b)
{
    // Start with unaugmented matrix not in REF
    if (Augmented == false && REF == false)
    {
        // Matrix must be square
        if (Rows == Cols)
        {
            // Augment matrix with column b
            augment(b);
            // Put in REF
            gaussElim();
            // Solution vector
            std::vector<float> xs(Rows, 0.0);
            // For each row, starting from last
            for (int i = Rows-1; i >= 0; i--)
            {
                float val = Vals[i][Cols-1];
                // If we're on the bottom row of the matrix...
                if (i == Rows-1)
                {
                    xs[Cols-2] = val / Vals[i][Cols - 1];
                }
                else // if (i < Rows-1)
                {
                    int xIdx = 0;
                    for (int j = Cols-2; j >= 0; j--)
                    {
                        val -= Vals[i][j] * xs[j];
                        if (Vals[i][j] != 0)
                        {
                            xIdx = j;
                        }
                    }
                    xs[xIdx] = val / Vals[i][xIdx];
                }
            }
            Matrix x = Matrix(1, (int)xs.size(), {xs});
            x.transpose();
            std::cout << "x contains: " << std::endl;
            x.printMatrix();
            return x;
        }
        // Matrix not square
        else if (Rows < Cols)
        {
            std::cerr << "Error: System is underdetermined. Matrix must be square.";
            throw "Underdetermined system error.";
        }
        else // if (Rows > Cols)
        {
            std::cerr << "Error: System is overdetermined. Matrix must be square.";
            throw "Overdetermined system error.";
        }
    }
    else
    {
        std::cerr << "Error: Please start with non-augmented and non-REF matrix.";
        throw "Matrix not in proper form error.";
    }    
}

float Matrix::trace()
{
    // Matrix must be square
    if (Rows == Cols)
    {
        if (Augmented == false)
        {
            float trace = 0;
            for (int i = 0; i < Rows; i++)
            {
                trace += Vals[i][i];
            }
            return trace;
        }
        else
        {
            std::cerr << "Error: Matrix must be non-augmented.";
            throw "Matrix not in proper form error.";
        }
    }
    else
    {
        std::cerr << "Error: Matrix must be square.";
        throw "Matrix not in proper form error.";
    }
}

// Implement using gaussian elimination to simplify
// Another way would be with LU decomposition
float Matrix::determinant()
{   
    // Matrix must be square
    if (Rows == Cols)
    {
        // Gaussian elim to get scaling factor
        // and make matrix upper triangular
        float scaling_factor = gaussElim();
        // Multiply diagonal elements
        for (int i = 0; i < Rows; i++)
        {
            scaling_factor *= Vals[i][i];
        }
        return scaling_factor;
    }
    else
    {
        std::cerr << "Error: Matrix must be square.";
        throw "Matrix not in proper form error.";
    }
}

// Helper functions for gram_schmidt()
float dot_product(std::vector<float> v1, std::vector<float> v2)
{
    // Vectors must be same length
    if ((int)v1.size() == (int)v2.size())
    {
        float sum = 0;
        for (int i = 0; i < (int)v1.size(); i++)
        {
            sum += v1[i] * v2[i];
        }
        return sum;
    }
    else
    {
        std::cerr << "Error: Vectors must be same length.";
        throw "Vectors not same length error.";
    }
}

float gs_coeff(std::vector<float> v1, std::vector<float> v2)
{
    return dot_product(v1, v2) / dot_product(v1, v1);
}

// Projection of v2 onto v1
std::vector<float> proj(std::vector<float> v1, std::vector<float> v2)
{
    std::vector<float> proj_vect;
    float coeff = gs_coeff(v1, v2);
    for (int i = 0; i < (int)v1.size(); i++)
    {
        proj_vect.push_back(coeff * v1[i]);
    }
    return proj_vect;
}

//std::vector<std::vector<float>> gram_schmidt(Matrix A)
std::vector<std::vector<float>> Matrix::gram_schmidt()
{
    // We want to feed gs algo the columns of the matrix < <row1>, <row2>, ... , <rown> >
    // so transpose first so we have < <col>, <col2>, ... , <coln> > that we can iterate over
    transpose();
    std::vector<std::vector<float>> qs; // Vector of orthonormal vectors to return
    // For each column in matrix
    for (std::vector<float> col : Vals)
    {
        std::vector<float> u = col;
        // For each vector in qs
        if (!qs.empty())
        {
            for (std::vector<float> q : qs)
            {
                std::vector<float> proj_vect = proj(q, u);
                for (int i = 0; i < (int)u.size(); i++)
                {
                    u[i] -= proj_vect[i];
                }
            }
        }
        // Append u to qs
        qs.push_back(u);
    }
    // Normalise vectors in qs
    for (int i = 0; i < (int)qs.size(); i++)
    {
        //std::vector<float> q = qs[i];
        float q_norm = std::sqrt(dot_product(qs[i], qs[i]));
        //std::cout << q_norm << std::endl;
        for (int j = 0; j < (int)qs[i].size(); j++)
        {
            //std::cout << j << ", " << qs[i][j] << std::endl;
            //float normalised = q[i] / q_norm;
            qs[i][j] /= q_norm;
        }
    }
    // Remember to transpose matrix back to original form
    transpose();
    // Return list of orthonormal vectors
    return qs;
}

// Helper functions for qr_decomp()
Matrix get_Q(Matrix A)
{
    std::vector<std::vector<float>> qs = A.gram_schmidt();
    Matrix Q = Matrix((int)qs.size(), (int)qs[0].size(), qs);
    Q.transpose();
    return Q;
}

Matrix get_R(Matrix A, Matrix Q)
{
    Q.transpose();
    Matrix R = Q * A;
    return R;
}

std::pair<Matrix, Matrix> Matrix::qr_decomp()
{
    Matrix Q = get_Q(*this);
    Matrix R = get_R(*this, Q);
    std::pair<Matrix, Matrix> QR_pair (Q, R);
    return QR_pair;
}

float sum_lower_triangular(Matrix A)
{
    float sum = 0;
    for (int i = 1; i < A.get_Rows(); i++)
    {
        for (int j = 0; j < i; j++)
        {
            sum += A.get_Vals()[i][j];
        }
    }
    return sum;
}

// Helper function checking that QR Algo is converging
// to an upper trianglular matrix
bool check_qr_convergence(Matrix Updated)
{
    float sum_updated = sum_lower_triangular(Updated);
    if (sum_updated < 1.0e-10)
    {
        return true;
    }
    else
    {
        return false;
    }
}

std::vector<float> get_eigenvalues(Matrix A)
{
    std::vector<float> eigenvalues;
    for (int i = 0; i < A.get_Rows(); i++)
    {
        eigenvalues.push_back(A.get_Vals()[i][i]);
    }
    return eigenvalues;
}

// Finds the eigenvalues and eigenvectors of a matrix.
// The elements of the returned vector are the eigenvalues
// (works for all matrices)
// The columns of the Q_final matrix are the associated eigenvectors
// (only works for symmetric matrices)
std::pair<std::vector<float>, Matrix> Matrix::qr_algo(int iterations)
{
    Matrix A = *this;
    Matrix Q_final = Matrix(Rows, Cols, 2); // Start with identity matrix of same size
    for (int i = 0; i < iterations; i++)
    {
        std::cout << "Iteration " << i << std::endl;
        std::pair<Matrix, Matrix> QR_pair = A.qr_decomp();
        Matrix Q = QR_pair.first;
        Matrix R = QR_pair.second;
        A = R * Q;
        Q_final = Q_final * Q;
        if (check_qr_convergence(A))
        {
            std::pair<std::vector<float>, Matrix> eigenstuff (get_eigenvalues(A), Q_final);
            return eigenstuff;
        }
    }
    std::cout << "Matrix didn't converge to upper triangular form." << std::endl;
    std::pair<std::vector<float>, Matrix> eigenstuff (get_eigenvalues(A), Q_final);
    return eigenstuff;
}

