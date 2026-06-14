\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{slc4eurosafe}[2021/11/23 A generic class for scilicium documents]

% Load color definitions (using scilicium colors)
\RequirePackage{sciliciumcolor}

% etoolbox provides additional commands for conditional formatting:
% \newbool
% \booltrue
% \boolfalse
% \ifbool
% \ifnumcomp comparison of **integer numbers**, which can be results of arbitrary arithmetic operations
% \newcounter, \setcounter to store numbers
% \ifnumcomp{6/2}{=}{5/2}{true}{false} <-- will be true, as we are using **integers**
\RequirePackage{etoolbox}

% Whether the language is English;
% defaults to true
\newbool{english}
\booltrue{english}

% Whether to include a field for ISBN where relevant (only public reports)
% defaults to false
\newbool{useISBN}

% Whether to use fontspec for font selection (XeTeX and LuaTeX)
% defaults to false
\newbool{fontspec}

% Whether the document is meant for printing, i.e. using CMYK.
% defaults to false, i.e. RGB display 
\newbool{print}

% Force use of Calibri font, defaults to false
\newbool{calibri}

%%% BEGIN Options section; no \RequirePackage or \LoadClass in here %%%

% Command to provide alternative translations in English and French
\newcommand{\EnglishFrench}[2]{\ifbool{english}{#1}{#2}}

% Classification options, internal is default as per scilicium policy
\newcommand*{\@classification}{\EnglishFrench{Internal}{Interne}}
\newcommand*{\@parentClass}{article}
\DeclareOption{unrestricted} {%
  \renewcommand*{\@classification}{\EnglishFrench{Unrestricted}{Tout public}}%
  \booltrue{useISBN}
}
\DeclareOption{internal}{%
  \renewcommand*{\@classification}{\EnglishFrench{Internal}{Interne}}%
}
\DeclareOption{restricted}{%
  \renewcommand*{\@classification}{\EnglishFrench{Restricted}{Restreint}}%
}
\DeclareOption{confidential}{%
  \renewcommand*{\@classification}{\EnglishFrench{Confidential}{Confidentiel}}%
}
\DeclareOption{report}{%
  \renewcommand{\@parentClass}{report}
}
\DeclareOption{article}{%
  \renewcommand{\@parentClass}{article}
}
\DeclareOption{english}  {\booltrue{english}}
\DeclareOption{francais}    {\boolfalse{english}}
\DeclareOption{twocolumn}{\OptionNotUsed}
\DeclareOption{display}  {\boolfalse{print}}
\DeclareOption{print}    {\booltrue{print}\PassOptionsToPackage{cmyk}{xcolor}}
\DeclareOption{calibri}  {\booltrue{calibri}\booltrue{fontspec}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\@parentClass}}
\ProcessOptions\relax

%%% END Options section %%%

\LoadClass[a4paper,11pt]{\@parentClass}

% Use a prettier date format, and adapt according to language.
\RequirePackage[useregional]{datetime2}
% provide \DTMnow


% If English is selected, prefer British, but have American ready.
\RequirePackage[\ifbool{english}{main=british,american}{}]{babel}

% Check what compiler is running so we know how to handle fonts later
\RequirePackage{iftex}
\ifXeTeX \booltrue{fontspec}\else\fi
\ifLuaTeX\booltrue{fontspec}\else\fi

% Set fonts.
% Carlito is a free clone of Calibri; "lining" option for "normal" digits.
% STIX 2 is an advanced open font, broadly compatible with Times New Roman. 
% it also has great math rendering.
\ifbool{fontspec}{ % XeLaTeX or LuaLaTeX, advanced font handling
  \RequirePackage{fontspec,unicode-math}
  \defaultfontfeatures{Ligatures=TeX}
  %\setmainfont{STIX Two Text}
  %\setmathfont{STIX Two Math}
  \setsansfont{\ifbool{calibri}{Calibri}{Carlito}}
}{ % Ye goode olde LaTeX and PDFLaTeX
  \RequirePackage[utf8]{inputenc}
  \RequirePackage[T1]{fontenc}
  %\RequirePackage{stix2}
  \RequirePackage[lining]{carlito}
}
% Use sans font by default
\renewcommand{\familydefault}{\sfdefault}

% Define \and function for multiple authors
\renewcommand{\and}{\\*}

% Define the main logo, whether CMYK (print) or RGB (screen)
\RequirePackage{graphicx}
\graphicspath{{./template/images/}}
% \newcommand{\@mainlogo}{SINTEF_Logo_Sentrert_\ifbool{print}{CMYK}{RGB}}
\newcommand{\@mainlogo}{Scilicium_Logo_centered_path_\ifbool{print}{CMYK}{RGB}}

\RequirePackage{template/sty/sciliciumlabels}
\InputIfFileExists{sciliciumconfig}{}{%
  \ClassWarning{sciliciumdoc}{%
  The sciliciumconfig.tex file is missing.\MessageBreak
  You can generate it by double-clicking sciliciumconfig.js., or by
  using the sciliciumconfig-example.tex as a template.}
}

% If some user has loaded hyperref,
% check for it and use the starred (non-linking) version of \pageref
% for page count (useless/ugly link)
\newcommand*\PlainPageRef[1]{%
  \@ifundefined{@pagerefstar}{\pageref{#1}}{\pageref*{#1}}%
}

% Information box
\newcommand{\@info}[1][sciliciumgreen]{%
  \noindent%
  \scriptsize%
  \raggedright%
  \setlength{\parskip}{1ex}%
  \color{#1}%
  \textbf{\@institute}

  \@addresslabel:\\\@address
  
  \@phonelabel: +33\,6\,40\,44\,19\,13

  \texttt{contact@scilicium.com}

  \@VATlabel:\\\@vat

}

% This separating line is used across several documents,
% for example in footers and title pages.
\RequirePackage{template/sty/sciliciumcolor}
\newcommand{\@separator}{%
  % To make sure we have spacing on both sides, make an invisible rule, 2X tall
  \rule{0ex}{2ex}%
   % Place the dashed rule 1X high
  \textcolor{sciliciumgray}{\rule[1ex]{\textwidth}{0.25pt}}%
}

\DeclareUnicodeCharacter{0391}{\ensuremath{\Alpha}}
\DeclareUnicodeCharacter{03B1}{\ensuremath{\alpha}}
\DeclareUnicodeCharacter{0392}{\ensuremath{\Beta}}
\DeclareUnicodeCharacter{03B2}{\ensuremath{\beta}}
\DeclareUnicodeCharacter{0393}{\ensuremath{\Gamma}}
\DeclareUnicodeCharacter{03B3}{\ensuremath{\gamma}}
\DeclareUnicodeCharacter{0394}{\ensuremath{\Delta}}
\DeclareUnicodeCharacter{03B4}{\ensuremath{\delta}}
\DeclareUnicodeCharacter{0395}{\ensuremath{\Epsilon}}
\DeclareUnicodeCharacter{03B5}{\ensuremath{\epsilon}}
\DeclareUnicodeCharacter{0396}{\ensuremath{\Zeta}}
\DeclareUnicodeCharacter{03B6}{\ensuremath{\zeta}}
\DeclareUnicodeCharacter{0397}{\ensuremath{\Eta}}
\DeclareUnicodeCharacter{03B7}{\ensuremath{\eta}}
\DeclareUnicodeCharacter{0398}{\ensuremath{\Theta}}
\DeclareUnicodeCharacter{03B8}{\ensuremath{\theta}}
\DeclareUnicodeCharacter{0399}{\ensuremath{\Iota}}
\DeclareUnicodeCharacter{03B9}{\ensuremath{\iota}}
\DeclareUnicodeCharacter{039A}{\ensuremath{\Kappa}}
\DeclareUnicodeCharacter{03BA}{\ensuremath{\kappa}}
\DeclareUnicodeCharacter{039B}{\ensuremath{\Lambda}}
\DeclareUnicodeCharacter{03BB}{\ensuremath{\lambda}}
\DeclareUnicodeCharacter{039C}{\ensuremath{\Mu}}
\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}
\DeclareUnicodeCharacter{039D}{\ensuremath{\Nu}}
\DeclareUnicodeCharacter{03BD}{\ensuremath{\nu}}
\DeclareUnicodeCharacter{039E}{\ensuremath{\Xi}}
\DeclareUnicodeCharacter{03BE}{\ensuremath{\xi}}
\DeclareUnicodeCharacter{039F}{\ensuremath{\Omicron}}
\DeclareUnicodeCharacter{03BF}{\ensuremath{\omicron}}
\DeclareUnicodeCharacter{03A0}{\ensuremath{\Pi}}
\DeclareUnicodeCharacter{03C0}{\ensuremath{\pi}}
\DeclareUnicodeCharacter{03A1}{\ensuremath{\Rho}}
\DeclareUnicodeCharacter{03C1}{\ensuremath{\rho}}
\DeclareUnicodeCharacter{03A3}{\ensuremath{\Sigma}}
\DeclareUnicodeCharacter{03C3}{\ensuremath{\sigma}}
\DeclareUnicodeCharacter{03A4}{\ensuremath{\Tau}}
\DeclareUnicodeCharacter{03C4}{\ensuremath{\tau}}
\DeclareUnicodeCharacter{03A5}{\ensuremath{\Upsilon}}
\DeclareUnicodeCharacter{03C5}{\ensuremath{\upsilon}}
\DeclareUnicodeCharacter{03A6}{\ensuremath{\Phi}}
\DeclareUnicodeCharacter{03C6}{\ensuremath{\varphi}}
\DeclareUnicodeCharacter{03A7}{\ensuremath{\Chi}}
\DeclareUnicodeCharacter{03C7}{\ensuremath{\chi}}
\DeclareUnicodeCharacter{03A8}{\ensuremath{\Psi}}
\DeclareUnicodeCharacter{03C8}{\ensuremath{\psi}}
\DeclareUnicodeCharacter{03A9}{\ensuremath{\Omega}}
\DeclareUnicodeCharacter{03C9}{\ensuremath{\omega}}


\RequirePackage{tcolorbox}
\newtcolorbox{gene_agent_box}{colback=sciliciumgreen!5!white,
colframe=sciliciumgreen}