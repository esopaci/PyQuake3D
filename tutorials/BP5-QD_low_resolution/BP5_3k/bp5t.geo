// Parameters
angle = 25;   // Dip angle of the fault in degrees
Len = 100;    // Fault length along x-axis
width = 40;   // Fault width along z-axis
lc = 1.5;     // Characteristic length for mesh cell size

// Define points for the rectangular fault plane
Point(1) = {-Len/2, 0, 0, lc};   // Bottom-left corner (A)
Point(2) = {Len/2, 0, 0, lc};    // Bottom-right corner (B)
Point(3) = {Len/2, 0, -width, lc}; // Top-right corner (C)
Point(4) = {-Len/2, 0, -width, lc}; // Top-left corner (D)
// Alternative points for a dipping fault (commented out)
//Point(3) = {Len/2, width * Cos(angle * Pi / 180), -width * Sin(angle * Pi / 180), lc};
//Point(4) = {-Len/2, width * Cos(angle * Pi / 180), -width * Sin(angle * Pi / 180), lc};

// Define lines connecting the points
Line(1) = {1, 2}; // Line from A to B (bottom edge)
Line(2) = {2, 3}; // Line from B to C (right edge)
Line(3) = {3, 4}; // Line from C to D (top edge)
Line(4) = {4, 1}; // Line from D to A (left edge)

// Define the surface by creating a line loop and plane
Line Loop(1) = {1, 2, 3, 4}; // Closed loop of lines A-B-C-D
Plane Surface(1) = {1};      // Create a planar surface from the line loop

// Mesh generation settings
//Mesh.Algorithm = 1;          // Mesh algorithm: 1 = Delaunay triangulation (triangular mesh)
//Mesh.Algorithm = 8;        // Alternative: 8 = Structured quadrilateral mesh (uncomment for quads)
//Mesh.ElementOrder = 1;       // Use first-order elements (linear triangles or quads)
//Mesh.RecombineAll = 1;       // Recombine triangles into quadrilaterals (enable for quad mesh)

// Generate the 2D mesh
Mesh 2;

// Save the mesh to a file
Mesh.Format = 2;             // Output format: 1 = MSH1 format (Gmsh legacy format)
//Save "fault_mesh.msh";       // Save the mesh as fault_mesh.msh