% Class for SINTEF reports with LaTeX.
%
% by Federico Zenith, federico.zenith@sintef.no.

\ProvidesClass{sciliciumtestreport}[2020/04/02 A class to prepare test reports in 
scilicium]

% Should not be used in reports
\DeclareOption{internal}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{sciliciumdoc}}
\ProcessOptions\relax

\LoadClass[report]{template/cls/sciliciumdoc}

% Set margins
\RequirePackage[top=3cm,          bottom=3cm,
                left=2cm,         right=2cm,
                headheight=2.51cm, headsep=0.5cm,
                footskip=1ex]{geometry}

% Definition of variables
\heading{\EnglishFrench{Test Report}{Rapport de test}}

\RequirePackage{fancyhdr,lastpage}
\pagestyle{fancy}

% Standard header & footer
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}% clear all header and footer fields
\cfoot{%
  \color{sciliciumgray}%
  \@separator\newline%
  \hspace*{0.3\textwidth}%
  \begin{minipage}[c]{0.55\textwidth}%
    \scriptsize\centering
    \EnglishFrench{The report is the client's property and cannot be given to a
                  third party without the client’s written consent.
                  The report shall not be reproduced except in full without the 
                  written approval of scilicium.}
                 {Ce rapport est la propriété du client et ne peut pas être transmis
                  a une tierce partie sans l'accord préalable écrit du client.
                  Ce rapport ne peut être reproduit que dans son intégralité, sauf accord
                  préalable écrit du client.}
  \end{minipage}%
  \hfill
  \thepage\ \EnglishFrench{of}{sur} \PlainPageRef{LastPage}%
  \hspace*{\tabcolsep}\newline%
  \@separator
}

% plain style is conjured by first pages of new chapters
% empty definition makes it a clone of fancy
\fancypagestyle{plain}{}

\fancypagestyle{coverpage}{%
  \fancyhf{}% clear all header and footer fields
  \lhead{\renewcommand{\baselinestretch}{0}%
         \raisebox{4.85mm}%
         {\includegraphics[height=3.62cm]{\@mainlogo}}%
        }
}

\fancypagestyle{firstpage}{%
  \lhead{\renewcommand{\baselinestretch}{0}%
         \raisebox{0.2mm}{\includegraphics[height=2.34cm]{\@mainlogo}}%
         }
}

% Title page, recipient address and information colophon
\RequirePackage{setspace,template/sty/sciliciumhistory,template/sty/sciliciumattachments}
\newcommand{\frontmatter}{%
  \thispagestyle{coverpage}%
  \newgeometry{top=4.75cm,        bottom=0cm,
               left=5.2cm,          right=0cm,
               headheight=4.25cm, headsep=5mm}
  \noindent%
  \textcolor{sciliciumblue}{\bfseries\@reportnumber\ - \@classification}\\
  \begin{minipage}[t]{13cm}
    {\fontsize{32}{32}\selectfont%
     \vspace{1.5ex}%
     \@heading\\
    }%
    \Huge       \vspace{2.5ex}\\\@title\\
    \normalsize \vspace{2ex}\\\@subtitle\\
                \vspace{2ex}\\\textbf{\@authorlabel}\\
                \@author
  \vspace{1ex}
  {\@titlefigure}
  \end{minipage}
  \hspace*{3mm}
  \addtocounter{page}{-1}%
  \restoregeometry%
% -------- END OF COVER PAGE --------

% -------- START FIRST PAGE --------
  \thispagestyle{firstpage}%
  \noindent%
  \begin{minipage}[t][0.99\textheight]{0.25\textwidth}
    \raggedright
    {\@info}
    \vspace{2ex}
    \@keywordlabel:\\
    \@keywords
        
    \vfill
    \includegraphics[width=\textwidth]{isoinscription}
  \end{minipage}
  \hfill
  \begin{minipage}[t][0.99\textheight]{0.7\textwidth}
    \setlength{\parskip}{2ex}%
    \raggedright%
    \Huge \@heading

    \vspace{2ex}
    \huge\textbf \@title\\
    \normalsize \@subtitle

    \vspace{4ex}%
    \noindent%
    \hspace*{\tabcolsep}
    \begin{minipage}{0.6\textwidth}
      \@labeltext \@versionlabel\\
      \@version
    \end{minipage}%
    \begin{minipage}{0.37\textwidth}
      \@labeltext \@datelabel\\
      \@date
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.97\textwidth}
      \@labeltext \@Authorlabel\\
      \@author
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.6\textwidth}
      \@labeltext \@clientlabel\\
      \@client
    \end{minipage}%
    \begin{minipage}{0.37\textwidth}
      \@labeltext \@clientreflabel\\
      \@clientref
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.6\textwidth}
      \@labeltext \@projectlabel\\
      \@project
    \end{minipage}%
    \begin{minipage}{0.37\textwidth}
      \@labeltext \@pageslabel\\
      \PlainPageRef{LastPage} \@attachmentpages
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.6\textwidth}
      \@labeltext \@testobjectlabel\\
      \@testobject
    \end{minipage}%
    \begin{minipage}{0.37\textwidth}
      \@labeltext \@datereceivedlabel\\
      \@datereceived
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.3\textwidth}
      \@labeltext \@testprogramlabel\\
      \@testprogram
    \end{minipage}%
    \begin{minipage}{0.3\textwidth}
      \@labeltext \@testlocationlabel\\
      \@testlocation
    \end{minipage}%
    \begin{minipage}{0.37\textwidth}
      \@labeltext \@testdatelabel\\
      \@testdate
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.97\textwidth}
      \@labeltext \@abstractlabel\\
      \@abstract
    \end{minipage}
    \vfill
    \begingroup
    \centering\footnotesize
    \EnglishFrench{The test results relate only to the items tested.}
                 {Les résultats de test ne portent que sur les échantillons testés.}\\
    \endgroup
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.5\textwidth}
      \@labeltext \@preparedlabel\\
      \@prepared
    \end{minipage}%
    \begin{minipage}{0.47\textwidth}
      \@labeltext\textmd\@signaturelabel\bigskip
    \end{minipage}\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.5\textwidth}
      \@labeltext \@approvedlabel\\
      \@approved
    \end{minipage}%
    \begin{minipage}{0.47\textwidth}
      \@labeltext\textmd\@signaturelabel\bigskip
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}{0.5\textwidth}
      \@labeltext \@reportnumberlabel\\
      \@reportnumber
    \end{minipage}%
    \begin{minipage}{0.47\textwidth}
      \@labeltext \@classificationlabel\\
      \@classification
    \end{minipage}\\    
  \end{minipage}

  \clearpage

  \historypage

  \tableofcontents

  \attachmentsection

  \clearpage

  \noindent
}
