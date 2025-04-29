## Intro to AI CS 830

This is an Intro to AI, CS 830, UNH.
This is an "implementation intensive" class.
Despite following 10 challenges this semester, I spent 5 to 8 hours per week for each one of these assignments, in CS 830 which is an implementation intensive" class.
Given the fast paced nature of the assignments, they are very basic and are not necessarily standardized code. Due to the following challenges listed, I was targeting for "meets expectations" as I had to focus on the following challenges consistently.

The assignments listed are already graded and do not disclose what the problem is.
They also do not disclose any assignment relevant details.

I have 10 challenges this semester Spring 2025 while I was enrolled in this class and ongoing:

- To find an Internship/Summer part-time.
- To keep up with Teaching Assistant works.
- To participate and present in 2 symposiums.
- To present a guest lecture for a class.
- To find funding for my PhD on my own and still exploring.
- To find an advisor for Natural Language Understanding.
- To withstand extreme extenuating circumstances, specially curated for me.
- Laptop port's was destroyed also had to rent a laptop. 
- Due to laptop charging port problem, Assignment 9 was fully written in a nano editor in like 7 hours, I was prepared I would not submit the assignment, but I did.

### Final project for CS 830

My course project is on First Order Logic, CNFs, Clausal forms and Resolution Refutation. That would be more elaborate. - by May 15th 2025.

### Note about assignment 6:
The grammar for CNF is custom and is unlike regular CNFs.

### Assignment 6's questions and Syllabus:

Syllabus says **prohibits the use of outside/external code, requires that work submitted is student's own"**. 
Assignment 6 provides an exception to this syllabus policy that allows the use of external parser generators like ANTLR or Bison, which generate code files considered **external code**, contradicting the syllabus policy's prohibition on external code. It also requires submitting these generated files and the parser grammar file "as is," further conflicting with the policy. Additionally, the provided parsers are unreliable due to line-ending sensitivity on Windows versus Linux, creating technical challenges that hinder compliances. However ANTLR and Bison do not have any boiler-plate grammars written by software developers, that work between windows and linux environments at the same time.

- <a href="https://github.com/antlr/antlr4">ANTLR4 version's original </a> which provides grammar files from <a href="https://github.com/antlr/grammars-v4">

- The proof that the developer-provide-default-example does not run on default ANTLR4 parser on linux <a href="https://github.com/antlr/grammars-v4/blob/master/fol/fol.g4">fol/fol.g4</a> as they have different line endings not honored on Linux but work on Windows. They fail to recognize certain patterns of words as names/variables/predicates/functions.

Due to this problem from ANTLR4 parsers, I implemented my own parser for assignment 7 which is available at: <a href="https://github.com/sushmaakoju/cs830/blob/master/sakoju-a07/code.py"> my own parser (recursive descent) </a>

### Assignment 10 problem and Syllabus exceptions:

Syllabus says **"You may use any programming language you wish in this course."**
But assignment 10 says "Python might not be the best choice for this assignment, as
runtime efficiency is important. It might be helpful to keep in mind that the number of
instances is much larger than the number of attributes which is larger than the number of
classes."

### Advice/suggestion/recommendations

As a Teaching Assistant/Instructor, when students do work within the scope of stated exceptions to syllabus policy in assignments, given the open-ended nature of the some problems in the assignment, it is easy to see that this remains within the scope and honors code. The proof that the developer-provide-default-example does not run on default ANTLR4 parser on linux <a href="https://github.com/antlr/grammars-v4/blob/master/fol/fol.g4">fol/cnf.g4</a> as they have different line endings not honored on Linux but work on Windows is proof enough that this is within the scope. The assignment does not provide example empty grammar templates with Linux-honored line endings for ANTLR/Bison etc that actually would have helped. Best solution: it maybe better to not include ANTLR/Bison/others in the future assignments if empty grammar templates with Linux-supported Line Endings specifications was NOT provided. By providing empty grammar templates with Linux-supported Line Endings specifications, this would reduce time spent in updating and adjusting grammars between two different operating systems. This also avoids looking into examples that are provided by ANTLR developers which they may have written on different Operating systems. This is also helpful for another reason that when the open-source parsers also undergo frequent changes subject to various other depending softwares such as Programming languages they were written in etc, which can impact example grammar templates provided by developers.
