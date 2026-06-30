% Class for Scilicium reports with LaTeX.
%
% by Théophile Grébert, theophile@scilicium.com.
%
% Based on a class for SINTEF reports with LaTeX, by Federico Zenith, federico.zenith@sintef.no, found on OverLeaf

\ProvidesClass{sciliciumreport}[2022/11/24 A class to prepare reports in scilicium style]

\newcommand*{\@summaryclassification}{\EnglishFrench{Restricted}{Restreint}}

\DeclareOption{unrestrictedsummary}{%%
  \renewcommand*{\@summaryclassification}{\EnglishFrench{Unrestricted}{Diffusion libre}}%
}
\DeclareOption{restrictedsummary}{%
  \renewcommand*{\@summaryclassification}{%
    \EnglishFrench{Restricted}{Restreint}
  }
}
\DeclareOption{confidentialsummary}{%
  \renewcommand*{\@summaryclassification}{%
    \EnglishFrench{Confidential}{Confidentiel}%
  }
}

% Should not be used in reports
\DeclareOption{internal}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{sciliciumdoc}}
\ProcessOptions\relax


% Setting up alternating colors for table rows
\PassOptionsToPackage{table}{xcolor}


\LoadClass[report]{template/cls/sciliciumdoc}

% Set margins
\RequirePackage[top=3cm,          bottom=3cm,
                left=2cm,         right=2cm,
                headheight=2.51cm, headsep=0.5cm,
                footskip=3ex]{geometry}

% Setting up link colors (in table of contents, and links in the document)
\RequirePackage{hyperref}

\hypersetup{
  pdftitle={\@pdftitle},
  pdfauthor={SciLicium},
  colorlinks=true,
  linkcolor={sciliciumdarkgreen},
  filecolor={maroon},
  citecolor={blue},
  urlcolor={blue},
  pdfcreator={SciLicium analysis pipeline}}

\newcommand{\@sponsorname}{}
\newcommand{\sponsorname}[1]{\renewcommand{\@sponsorname}{#1}}
\newcommand{\@sponsorcontact}{}
\newcommand{\sponsorcontact}[1]{\renewcommand{\@sponsorcontact}{#1}}
\newcommand{\@sponsoremail}{}
\newcommand{\sponsoremail}[1]{\renewcommand{\@sponsoremail}{#1}}
\newcommand{\@sponsoraddress}{}
\newcommand{\sponsoraddress}[1]{\renewcommand{\@sponsoraddress}{#1}}

\newcommand{\@testfacilityname}{}
\newcommand{\testfacilityname}[1]{\renewcommand{\@testfacilityname}{#1}}
\newcommand{\@testfacilitycontact}{}
\newcommand{\testfacilitycontact}[1]{\renewcommand{\@testfacilitycontact}{#1}}
\newcommand{\@testfacilityemail}{}
\newcommand{\testfacilityemail}[1]{\renewcommand{\@testfacilityemail}{#1}}
\newcommand{\@testfacilityaddress}{}
\newcommand{\testfacilityaddress}[1]{\renewcommand{\@testfacilityaddress}{#1}}

\newcommand{\@testsitename}{}
\newcommand{\testsitename}[1]{\renewcommand{\@testsitename}{#1}}
\newcommand{\@testsitecontact}{}
\newcommand{\testsitecontact}[1]{\renewcommand{\@testsitecontact}{#1}}
\newcommand{\@testsiteemail}{}
\newcommand{\testsiteemail}[1]{\renewcommand{\@testsiteemail}{#1}}
\newcommand{\@testsiteaddress}{}
\newcommand{\testsiteaddress}[1]{\renewcommand{\@testsiteaddress}{#1}}

\newcommand{\@sponsorlabel}{\EnglishFrench{Sponsor}{Sponsor}}
\newcommand{\@testfacilitylabel}{\EnglishFrench{Test Facility}{Installation d'Essai}}
\newcommand{\@testsitelabel}{\EnglishFrench{Test Site}{Site d'Essai}}


% Setting up headings/section colors
\RequirePackage{sectsty}% http://ctan.org/pkg/sectsty
\sectionfont{\color{sciliciumpoliceblue}}
\subsectionfont{\color{sciliciumsteelblue}}
\subsubsectionfont{\color{sciliciumpearlviolet}}
% \sectionfont{\color{sciliciumsteelblue}}
% \subsectionfont{\color{sciliciumdarksteelblue}}
% \subsubsectionfont{\color{sciliciumdarkgrayblue}}
% \sectionfont{\color{sciliciumpoliceblue}}
% \subsectionfont{\color{sciliciumsky}}
% \subsubsectionfont{\color{sciliciumpearlviolet}}



% Setting up headers and footers
\RequirePackage{fancyhdr,lastpage}
\pagestyle{fancy}

% Standard header & footer
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}% clear all header and footer fields
\lhead{\renewcommand{\baselinestretch}{0}%
       \raisebox{0.2mm}{\includegraphics[height=2.34cm]{\@mainlogo}}%
       }
\cfoot{%
  \color{sciliciumgreen}%
  \@separator\newline%
    \hfill
    \begin{minipage}[c]{0.25\textwidth}
        \@LabelText \@projectlabel\newline
        \@clientref
    \end{minipage}%
    \begin{minipage}[c]{0.25\textwidth}
    \@LabelText \@reportnumberlabel\newline
    \@reportnumber
  \end{minipage}%
  \begin{minipage}[c]{0.25\textwidth}
    \@LabelText \@versionlabel\newline
    \@version
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
         \hspace*{2.3cm}\raisebox{4.85mm}%
         {\includegraphics[height=3.62cm]{\@mainlogo}}%
        }
  \rhead{}
  \chead{}
}

\fancypagestyle{firstpage}{%
  \fancyhf{}
  \fancyhead{}
  \fancyfoot{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{\renewcommand{\baselinestretch}{0}%
       \raisebox{0.2mm}{\includegraphics[height=2.34cm]{\@mainlogo}}%
       }
  \cfoot{%
    \color{sciliciumgreen}
    \@separator\newline
    \begin{minipage}[c]{0.4\textwidth}
    ~\\\phantom{opera}
    \end{minipage}%
    \hfill
    \begin{minipage}[t]{0.4\textwidth}
      \flushright%
      \thepage\ \EnglishFrench{of}{sur} \PlainPageRef{LastPage}
    \end{minipage}%
    \hspace*{\tabcolsep}\newline%
    \@separator
  }
}

% Logo used on the coloured back cover. Overridable for custom branding.
\newcommand{\@invertedlogo}{Scilicium_Logo_centered_inverted_\ifbool{print}{CMYK}{RGB}}

\fancypagestyle{backcover}{%
  \fancyhf{}% clear all header and footer fields
  \lhead{\renewcommand{\baselinestretch}{0}%
         \raisebox{0.2mm}{\includegraphics[height=2.34cm]{\@invertedlogo}}%
        }
}

\newcommand{\@coverbackground}{}
\newcommand{\coverbackground}[1]{\renewcommand{\@coverbackground}{#1}}

\newcommand{\@clientlogo}{}
\newcommand{\clientlogo}[1]{\renewcommand{\@clientlogo}{#1}}

% Contact details (used by the alternative "contact" back page). Empty by default.
\newcommand{\@contactemail}{}
\newcommand{\contactemail}[1]{\renewcommand{\@contactemail}{#1}}
\newcommand{\@contactphone}{}
\newcommand{\contactphone}[1]{\renewcommand{\@contactphone}{#1}}




% Title page, first page, table of contents etc.
\RequirePackage{eso-pic,template/sty/sciliciumhistory,template/sty/sciliciumattachments,tikz}
\newcommand{\frontmatter}{%
% -------- START FIRST PAGE (ancienne page 2) --------
  \thispagestyle{firstpage}%
  \noindent%
  \begin{minipage}[t][0.92\textheight]{0.3\textwidth}
    \flushleft{\small\@info}
    \vfill
  \end{minipage}%
  \begin{minipage}[t][0.92\textheight]{0.7\textwidth}
    \setlength{\parskip}{2ex}%
    \raggedright%
    \Huge\color{sciliciumpoliceblue}\@clientref

    \vspace{2ex}
    \huge\color{sciliciumpoliceblue}\@title
    \normalcolor\normalsize

    %\normalsize \@subtitle

    \vspace{4ex}%
    \noindent%
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@versionlabel\\
      \@version
    \end{minipage}%
    \begin{minipage}[t]{0.47\textwidth}
      \@LabelText \@datelabel\\
      \@date
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.5\textwidth}
      \@LabelText \@Authorlabel\\
      \@author
    \end{minipage}%
    \begin{minipage}[t]{0.47\textwidth}
      \@LabelText \@projectlabel\\
      \@reportnumber
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.97\textwidth}
      \@LabelText \@pageslabel\\
      \PlainPageRef{LastPage}
      \@attachmentpages
    \end{minipage}\\
    \@separator\\

    % Informations en lignes: Sponsor / Test Facility / Test Site
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.97\textwidth}
      \@LabelText \@sponsorlabel\\
      \ifdefempty{\@sponsorname}{N/A}{%
        \@sponsorname%
        \ifdefempty{\@sponsorcontact}{}{\newline Contact: \@sponsorcontact}%
        \ifdefempty{\@sponsoremail}{}{\newline Email: \@sponsoremail}%
        \ifdefempty{\@sponsoraddress}{}{\newline \@sponsoraddress}%
      }
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.97\textwidth}
      \@LabelText \@testfacilitylabel\\
      \ifdefempty{\@testfacilityname}{N/A}{%
        \@testfacilityname%
        \ifdefempty{\@testfacilitycontact}{}{\newline Contact: \@testfacilitycontact}%
        \ifdefempty{\@testfacilityemail}{}{\newline Email: \@testfacilityemail}%
        \ifdefempty{\@testfacilityaddress}{}{\newline \@testfacilityaddress}%
      }
    \end{minipage}\\
    \@separator\\
    \hspace*{\tabcolsep}
    \begin{minipage}[t]{0.97\textwidth}
      \@LabelText \@testsitelabel\\
      \ifdefempty{\@testsitename}{N/A}{%
        \@testsitename%
        \ifdefempty{\@testsitecontact}{}{\newline Contact: \@testsitecontact}%
        \ifdefempty{\@testsiteemail}{}{\newline Email: \@testsiteemail}%
        \ifdefempty{\@testsiteaddress}{}{\newline \@testsiteaddress}%
      }
    \end{minipage}\\
    \@separator\\

    \vfill

    % Signatures agrandies pour 3 entités
    \@separator\\
    \hspace*{\tabcolsep}
	\begin{minipage}[t]{0.31\textwidth}
	  \@LabelText \textbf{\@approvedlabel}\\
      \@LabelText \textbf{(\@testsitelabel)}\\
      \vspace{2.5cm}
      \hrule
      \smallskip
      {\small Date and Signature}
	\end{minipage}%
	\begin{minipage}[t]{0.31\textwidth}
	  \@LabelText \textbf{\@approvedlabel}\\
      \@LabelText \textbf{(\@testfacilitylabel)}\\
      \vspace{2.5cm}
      \hrule
      \smallskip
      {\small Date and Signature}
	\end{minipage}%
	\begin{minipage}[t]{0.31\textwidth}
	\end{minipage}\\
	\@separator\\
  \end{minipage}


  \clearpage

  % \historypage

  \tableofcontents

  \clearpage
  \null
  \thispagestyle{empty}
  \clearpage
  \null
  \thispagestyle{empty}
  \clearpage

  % \attachmentsection  % Commented out - not needed for BRB-seq reports

  \noindent
}

\newcommand{\backmatter}{
  \clearpage%
  % Subtract one page so LastPage seems to refer to our intended last page.
  \addtocounter{page}{-1}%
  \thispagestyle{backcover}
  \pagecolor{sciliciumgreen}%
  \vspace*{\fill}
  \centering\Huge\bfseries\color{white}\@motto
  \vspace*{\fill}
}

% ---- Alternative SIMPLE first page: compact title + minimal info + TOC ----
\newcommand{\frontmattersimple}{%
  \thispagestyle{firstpage}%
  \vspace*{2cm}
  \begin{center}
    {\Huge\color{sciliciumpoliceblue}\@title\par}
    \vspace{1ex}
    {\large \@subtitle\par}
    \vspace{2ex}
    {\large \@LabelText \@projectlabel: \@reportnumber\par}
  \end{center}
  \vfill
  \noindent
  \begin{minipage}[t]{0.5\textwidth}
    \@LabelText \@versionlabel\\ \@version
  \end{minipage}%
  \begin{minipage}[t]{0.47\textwidth}
    \@LabelText \@datelabel\\ \@date
  \end{minipage}\\[1.5ex]
  \@separator\\
  \begin{minipage}[t]{0.97\textwidth}
    \@LabelText \@Authorlabel\\ \@author
  \end{minipage}\\
  \@separator\\
  \clearpage
  \tableofcontents
  \clearpage
  \noindent
}

% ---- Alternative COVER first page: full-page coloured cover, then detailed ----
% Avoid the `titlepage` environment (it resets the page counter, breaking the
% "page X of Y" footer). Use a plain styled page instead.
\newcommand{\frontmattercover}{%
  \thispagestyle{empty}%
  \pagecolor{sciliciumpoliceblue}\color{white}%
  \begin{center}
    \vspace*{\fill}
    \includegraphics[height=3.5cm,keepaspectratio]{\@invertedlogo}\\[1.5cm]
    {\Huge\bfseries \@title\par}\vspace{0.6cm}
    {\Large \@subtitle\par}
    \vspace*{\fill}
  \end{center}
  \clearpage
  \nopagecolor\color{black}%
  \frontmatter
}

% ---- Alternative CONTACT back page: plain logo + institute + contact ----
\newcommand{\backmattercontact}{%
  \clearpage%
  \addtocounter{page}{-1}%
  \thispagestyle{firstpage}%
  \vspace*{3cm}
  \begin{center}
    \includegraphics[height=3cm,keepaspectratio]{\@mainlogo}\\[1cm]
    {\Large\color{sciliciumpoliceblue}\@institute\par}
    \vspace{0.5ex}
    {\@address\par}
    \vspace{1ex}
    \ifdefempty{\@contactemail}{}{\texttt{\@contactemail}\par}
    \ifdefempty{\@contactphone}{}{\@contactphone\par}
  \end{center}
}
