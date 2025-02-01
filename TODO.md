# task steps to achive better parallelization
1. partition problem into tasks
2. identify communication chanells between tasks
3. aggregate tasks into composite tasks
4. map composite tasks into cores.

# General tasks
- generate pseudocode of parallelized implementation


# Improvements
- reading user input
- divide the load on the collector node
    * use split communications -> tree structured or MPI reduce
- consider multiple problem sizes(lec 13 part-II)


# Project Structure
## Introduction
- An accurate description of the assigned project should be provided, including analysis of the sequential algorithm that solves the problem addressed in the project.
- Pseudo-code, examples, graphs, figures, application instances etc. may be provided.
## Parallel design
- Preliminary study about the opportunities for parallelism inherent in the problem sequential algorithm. 
- State of the art analysis should be performed on parallel design strategies
- Alternative designs, related to different parallelization strategies, should be considered and discussed at this stage, appropriately motivating why some operations lend themselves to effective parallelization and others do not as well as why some data structures may or may not minimize the burden of communication and synchronization. 
- Reference to prior work as well as link with related work must be clearly stated in the report.
- Hybrid parallelization strategies, with data dependencies must be discussed too

## Implementation
- C implementation (or, possibly, C++). The code must be properly commented. In case the student identifies multiple parallelization strategies, several implementations can be provided discussing pros and cons of each strategy. The report can include some code snippets related to the most critical and interesting parts.
- A link to the repo must be provided too.
- Hybrid parallelization is recommended though it is not mandatory
## Performance and scalability analysis
- The student must analyze the performance of the developed implementation in terms of  execution time, speedup, and efficiency. 
- Both strong scalability and weak scalability should be evaluated where possible. 


# EValuations
## The evaluation will also take into account: 
- Clarity and effectiveness of presentation; 
- Depth, correctness and originality of theoretical analysis; 
- Technical skills and documentation of implementation; 
- Number and quality of multiple parallel strategies, if any; 
- Critical thinking in performance evaluation and analysis; 
## Significance of results: 
- since parallel programming is primarily concerned with performance, a good project must also provide a significant speedup.
## Note: 
- The report should be as concise as possible, but without detriment to clarity of presentation. Expected size: 6-10 pages
