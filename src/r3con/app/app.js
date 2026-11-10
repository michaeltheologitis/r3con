(() => {
  'use strict';

  /** @typedef {`r${number}`} RoundKind Relevance round k's kind, `r<k>`, its nodes `r<k>-<d>`, one per document d. */
  /** @typedef {'doc' | RoundKind | 'schema' | 'parsed' | 'answer'} NodeKind */
  /** @typedef {RoundKind | 'parsed'} PerDocumentKind */
  /** @typedef {Exclude<NodeKind, 'doc'>} StageKind */
  /** @typedef {'flow' | 'cite' | 'self' | 'context'} EdgeKind */
  /** @typedef {{ x: number, y: number }} Point */
  /** @typedef {{ x: number, y: number, k: number }} View */
  /** @typedef {{ left: number, top: number }} Offset */
  /** @typedef {{ dist: number, mid: Point, view: View }} Pinch */
  /** @typedef {(i: number, file: LoadedFile) => Point} Origin */
  // A file's size is the file's; its bytes are its text's, in UTF-8, which is what r3con is sent.
  /** @typedef {{ name: string, text: string, size: number, bytes: number, sample: boolean, uid: number }} LoadedFile */
  /** @typedef {{ files: LoadedFile[], at: Map<string, Point> }} Removed What remove all took: the files, and where their nodes were. */
  /** @typedef {{ removed?: Removed, at?: Map<string, Point> }} FilesChange */
  /** @typedef {{ text: string } | { why: string }} FileRead A file's text, or why it can't be added. */
  /** @typedef {{ name: string, size: number } & FileRead} DocumentRead */
  /**
   * A file's documents, many for a zip, and how many files it left out by their suffix.
   * @typedef {{ docs: DocumentRead[], skipped: number }} Unpacked
   */
  /**
   * Files to add; how many files in their folders were left out by their suffix, and those the browser couldn't read.
   * @typedef {{ files: File[], skipped: number, failed: string[] }} Picked
   */
  /**
   * A sentence a summary quotes from its document, and where it is in the document's text, or null where r3con didn't
   * find it. r3con counts code points, as Python indexes text; a run on the page holds UTF-16 code units, as a string
   * here is indexed, while SAMPLE and an event hold r3con's own.
   * @typedef {{ text: string, start: number | null, end: number | null }} Quote
   */
  /** @typedef {Quote & { snippet: string }} RecordedQuote A quote, and the id of the snippet quoting it. */
  /** @typedef {Quote & { summary: string }} RunQuote A quote, and the trail node of the summary quoting it. */
  /** @typedef {{ summary: string, provenance: number[], quotes: RecordedQuote[] }} RecordedSummary */
  /** @typedef {{ summary: string, provenance: number[], quotes: RunQuote[] }} Summary */
  /** @typedef {{ start: number, end: number }} TextRange A part of a document's text, in UTF-16 code units. */
  /** @typedef {TextRange & { by: string[] }} Span A part of a document's text that the summaries `by` quote. */
  /** @typedef {{ spans: Span[], sentences: TextRange[] }} Quoted What a trail quotes of one document. */
  /** @typedef {[string, string, 'cites' | 'summarizes' | 'parses']} TrailEdge `[from, to, kind]`. */
  /**
   * An answer's trail as r3con gives it (r3con.provenance): the nodes its walk under `rule` reached from the answer,
   * `answer`, `summary:<round>:<id>`, `parse:<id>` and `document:<id>`, where an id is a document's, or `<id>.<k>` for
   * part k of a document read in parts; and the edges it followed.
   * @typedef {{ rule: string, nodes: string[], edges: TrailEdge[] }} ProvenanceTrail
   */
  /** @typedef {{ document: string, start: number, end: number, text: string, name: string }} NamedQuote A quote behind an answer, as r3con gives it. */
  /**
   * The lit answer's trail as the page draws it: its nodes and lines, as the page names them, a summary of a document
   * read in parts as its document's; each summary on it, with the first node on it that cites the summary; the trail
   * nodes of the summaries whose quotes it carries; and what they quote of each document, by index.
   * @typedef {{ nodes: Set<string>, edges: Set<string>, citedBy: Map<string, string>, quoting: Set<string>, quoted: Quoted[] }} Trail
   */
  /** @typedef {{ parse: Record<string, unknown[]>, provenance: number[] }} Parse */
  /** @typedef {{ code: string, thought: string }} Schema */
  /**
   * An answer's provenance: the documents whose final-round summaries and parses it cites, by index, and its trail.
   * @typedef {{ summary_ids: number[], parse_ids: number[], trail: ProvenanceTrail }} Provenance
   */
  /**
   * An answer's provenance as r3con gives it (r3con.pipeline.Answer.provenance), citing by id.
   * @typedef {{ summary_ids: string[], parse_ids: string[], quotes: NamedQuote[], trail: ProvenanceTrail }} R3conProvenance
   */
  /** @typedef {{ step: number, tokens: number, usd: number }} Call */
  /** @typedef {{ tokens?: number | null, usd?: number | null }} Totals */
  /**
   * The run's own answer or an ask's; an ask also has its source and totals, and live, the server's id
   * for it. The text and provenance are null until it lands.
   * @typedef {object} Answer
   * @property {string} question
   * @property {string} model
   * @property {string | null} [endpoint]
   * @property {string | null} text
   * @property {Provenance | null} provenance
   * @property {'recorded' | 'simulated' | 'live'} [source]
   * @property {Totals | null} [totals]
   * @property {string | null} [id]
   */
  /**
   * A recorded, simulated or live run; answers[0] is its own, the rest asks over it. A live run has the server's id
   * for it. Its document ids are r3con's, or made up for a simulated run. Each relevance round's summaries are in
   * `rounds`, round k's at k - 1, so it has as many rounds as that has entries.
   * @typedef {object} Run
   * @property {boolean} simulated
   * @property {LoadedFile[]} docs
   * @property {Summary[][]} rounds
   * @property {Schema | null} schema
   * @property {Parse[]} parsed
   * @property {Answer[]} answers
   * @property {Record<string, Call> | null} calls
   * @property {Totals | null} totals
   * @property {string} [source]
   * @property {string | null} [id]
   * @property {string[]} ids
   * @property {Map<string, number>} indexes each document id's index
   * @property {string | null} [lit]
   */
  /**
   * @typedef {object} RecordedRun
   * @property {string} question
   * @property {string} model
   * @property {string} version
   * @property {string} runFolder
   * @property {string} created
   * @property {{ name: string, text: string, size: number }[]} docs
   * @property {string[]} ids
   * @property {RecordedSummary[][]} rounds
   * @property {Schema} schema
   * @property {Parse[]} parsed
   * @property {{ text: string, provenance: R3conProvenance }} answer
   * @property {Record<string, Call>} calls
   * @property {Totals} totals
   */
  /**
   * @typedef {{ question: string, model: string, folder: string, created: string, text: string,
   *   provenance: R3conProvenance, totals: Totals }} RecordedAsk
   */
  /** @typedef {{ id: string, kind: NodeKind, file: LoadedFile | null, x: number, y: number, el: HTMLElement, fold?: FoldRow }} PanelNode A fold's node has its fold. */
  /** @typedef {{ key: string, from: string, to: string, kind: EdgeKind, el: SVGPathElement }} PanelEdge */
  /** @typedef {[string, string, EdgeKind, ('draw' | 'fade')?]} Line A line to draw: from, to, its kind, and how it enters. */
  /** @typedef {'working' | 'done' | 'failed'} CallState */
  /** @typedef {{ start: number, end: number, open: boolean }} FoldRow Documents start to end, inclusive, as one row, or opened above their own rows. */
  /** @typedef {FoldRow | { doc: number }} PanelRow */
  /**
   * How a tab of more than FOLD_FROM_DOCS documents folds them.
   * @typedef {object} Folding
   * @property {LoadedFile[]} files the tab's files, or its run's documents
   * @property {Map<string, number>} index each document node's document
   * @property {Map<string, CallState>} calls each call by its node's id; a document node's is reading the file
   * @property {Map<NodeKind, number>} settled how many calls of each kind are back
   * @property {Set<number>} relevant documents with a snippet, or on an answer's trail, which have a row of their own
   * @property {Set<number>} kept documents kept in view from before the run until their first round is back
   * @property {Set<number>} opened documents in an opened fold
   * @property {boolean} stopped whether the run stopped before reading them all
   * @property {PanelRow[]} rows top to bottom
   * @property {number} closedRows how many rows there would be with every fold closed
   * @property {(FoldRow | null)[]} foldOf each document's fold, while it is closed
   * @property {number} frame the animation frame its rows are next drawn in, if they are waiting to be
   */
  /**
   * @typedef {object} Tab
   * @property {number} id
   * @property {LoadedFile[]} files
   * @property {Removed | null} removed while undo can put it back
   * @property {number} reading how many batches of files are being read for it, or wait to be
   * @property {string} filesSaid
   * @property {string} filesError
   * @property {string} draft
   * @property {Promise<void>} adding
   * @property {HTMLButtonElement} button
   * @property {HTMLButtonElement} close
   * @property {Run | null} run
   * @property {HTMLElement} chat
   * @property {boolean} chatToEnd
   * @property {HTMLElement | null} chatAnswer the first line of the answer landed since the chat last scrolled
   * @property {HTMLElement | null} turn
   * @property {(string | Node)[]} status
   * @property {HTMLElement} drawing
   * @property {{ base: SVGGElement, mid: SVGGElement, lit: SVGGElement }} layers
   * @property {HTMLElement} nodesEl
   * @property {Map<string, PanelNode>} nodes
   * @property {Map<string, PanelEdge>} edges
   * @property {Map<string, Point>} layout
   * @property {View} view
   * @property {boolean} userMoved
   * @property {number[]} timers
   * @property {Narrator | null} narr
   * @property {number} arrangeFrame
   * @property {Trail | null} trail the lit answer's, while the panel has one lit
   * @property {Folding | null} fold while it has more than FOLD_FROM_DOCS documents
   * @property {Set<string>} placed the nodes the user dragged since the panel was last laid out
   */
  /** @typedef {Tab & { run: Run, turn: HTMLElement }} RunTab A tab with a run, the only kind with nodes other than documents. */
  /** @typedef {ReturnType<typeof narrator>} Narrator */
  /** @typedef {{ kind: 'run' | 'ask', tab: Tab, job: Run | Answer, stopping: boolean }} Stream */
  /**
   * An event r3con app streams, with the fields this page reads (r3con.pipeline.run_pipeline lists
   * them all).
   * @typedef {{ type: 'run_started', rounds: number, document_ids: string[] }
   *   | { type: 'relevance_started', round: number, doc_id: string }
   *   | { type: 'relevance_finished', round: number, doc_id: string, part: number | null, summary: string, quotes: Quote[],
   *       provenance: string[] }
   *   | { type: 'schema_started' | 'reasoning_started' | 'ask_started' }
   *   | { type: 'schema_finished', schema_code: string, thought: string }
   *   | { type: 'parsing_started', doc_id: string }
   *   | { type: 'parsing_finished', doc_id: string, parse: Record<string, unknown[]>, provenance: string[] }
   *   | { type: 'reasoning_finished', answer: string,
   *       provenance: R3conProvenance }
   *   | { type: 'run_finished' | 'ask_finished', tokens: { total: number } | null, usd: number | null }
   *   | { type: 'run_failed' | 'ask_failed', stage: string | null, error: string }
   * } ServerEvent
   */
  /**
   * A provider's key as r3con app answers it: stored there, with its last characters as a hint, or found in an
   * environment variable. `other` lists the servers stored there, each its endpoint and its key's hint or null.
   * @typedef {{ source: 'here', hint: string } | { source: 'environment', variable: string }} KeyState
   * @typedef {{ endpoint: string, key: string | null }} Server
   * @typedef {Record<string, KeyState | Server[] | null>} KeyStatus
   */
  /** @typedef {{ model: string, endpoint?: string }} SavedModel A model saved to the menu, and its server, if any. */
  /** @typedef {{ node?: string | null, glyph?: string | null, dim?: boolean }} LineOptions */
  /** @typedef {{ outcome: string, error: string }} Ending How a run or an ask stopped or failed, and the error if it failed. */
  /** @typedef {Node | string | null | undefined | false | Child[]} Child */
  /** @typedef {Record<string, string | number | boolean | null | undefined | EventListener>} Props */
  /**
   * What differs between the page served by r3con app and the prototype.
   * @typedef {object} Mode
   * @property {string} note
   * @property {string} keysNote
   * @property {(question: string) => string} nextRun
   * @property {(question: string, asks: boolean) => boolean} needsModel
   * @property {(files: LoadedFile[]) => string | null} refuseFiles
   * @property {(file: File, noText: string) => Promise<FileRead>} extract
   * @property {(file: File) => Promise<Unpacked>} unzip
   * @property {(question: string, files: LoadedFile[]) => Run} newRun
   * @property {(tab: RunTab, narr: Narrator) => void} play
   * @property {(tab: Tab) => boolean} runGoing
   * @property {(tab: Tab) => void} stop
   * @property {(tab: RunTab, question: string) => void} ask
   * @property {(provider: string, key: string) => Promise<KeyStatus>} saveKey
   * @property {(endpoint: string, key: string) => Promise<KeyStatus>} saveServer
   * @property {(provider: string) => Promise<KeyStatus>} forgetKey
   * @property {(endpoint: string) => Promise<KeyStatus>} forgetServer
   * @property {() => Promise<KeyStatus>} readKeys
   */

  // SAMPLE and RECORDED_ASKS are generated by tests/app/build_sample.py; don't edit them by hand.
  /** @type {RecordedRun} */
  const SAMPLE = {
 "question": "Which maintenance contractor's sites logged the most equipment incidents in Q3 in total, and how many?",
 "model": "gpt-6-luna",
 "version": "0.6.4",
 "runFolder": "20261105T173546Z_bb4f0c37",
 "created": "2026-11-05T17:35:46+00:00",
 "docs": [
  {
   "name": "01_northgate.txt",
   "text": "MEMO — Northgate site, Q3 operations review\n\nEquipment incidents logged this quarter: 5.\nThree were conveyor stoppages, two were HVAC trips.\nPlanned maintenance at this site is carried out under contractor code CT-118.\nSite manager: Dana Reyes.\n",
   "size": 247
  },
  {
   "name": "02_eastfield.txt",
   "text": "MEMO — Eastfield site, Q3 operations review\n\nEquipment incidents logged this quarter: 9.\nMostly compressor faults, clustered in August.\nPlanned maintenance at this site is carried out under contractor code CT-204.\nSite manager: Sam Okafor.\n",
   "size": 242
  },
  {
   "name": "03_riverside.txt",
   "text": "MEMO — Riverside site, Q3 operations review\n\nEquipment incidents logged this quarter: 6.\nFour were electrical, two were water ingress after the storm in September.\nPlanned maintenance at this site is carried out under contractor code CT-118.\nSite manager: Priya Raman.\n",
   "size": 271
  },
  {
   "name": "04_contractor_registry.txt",
   "text": "APPROVED CONTRACTOR REGISTRY (extract, revised July)\n\nCT-118 .... Halloran Services Ltd — mechanical and electrical maintenance\nCT-204 .... Merrow Facilities Group — HVAC and compressed air\nCT-311 .... Bellweather Grounds — landscaping (seasonal)\n\nCodes are used in site paperwork; the registry is the authority for the mapping.\n",
   "size": 335
  },
  {
   "name": "05_cafeteria.txt",
   "text": "MEMO — Cafeteria refurbishment, phase 2\n\nThe serving counter will be replaced during the October shutdown. Seating\ncapacity rises from 60 to 82. Catering staff should expect two weeks of\nreduced service. No impact on production areas.\n",
   "size": 237
  }
 ],
 "ids": [
  "17f027",
  "964bc2",
  "9e02b3",
  "d9add2",
  "7109a9"
 ],
 "rounds": [
  [
   {
    "summary": "A Q3 operations-review memo for the Northgate site. It reports 5 equipment incidents and identifies contractor code CT-118 as responsible for planned maintenance at the site, contributing Northgate’s incident count to the contractor comparison.",
    "provenance": [],
    "quotes": [
     {
      "text": "Equipment incidents logged this quarter: 5.",
      "start": 45,
      "end": 88,
      "snippet": "17f027"
     },
     {
      "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
      "start": 141,
      "end": 218,
      "snippet": "17f027"
     }
    ]
   },
   {
    "summary": "A Q3 operations review memo for the Eastfield site. It reports 9 equipment incidents and identifies maintenance contractor code CT-204; this is one site's contribution to the overall contractor comparison.",
    "provenance": [],
    "quotes": [
     {
      "text": "Equipment incidents logged this quarter: 9.",
      "start": 45,
      "end": 88,
      "snippet": "964bc2"
     },
     {
      "text": "Planned maintenance at this site is carried out under contractor code CT-204.",
      "start": 136,
      "end": 213,
      "snippet": "964bc2"
     }
    ]
   },
   {
    "summary": "A Q3 operations review memo for the Riverside site. It reports 6 equipment incidents and identifies maintenance contractor code CT-118, contributing a total of 6 incidents for that contractor in this document.",
    "provenance": [],
    "quotes": [
     {
      "text": "Equipment incidents logged this quarter: 6.",
      "start": 45,
      "end": 88,
      "snippet": "9e02b3"
     },
     {
      "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
      "start": 164,
      "end": 241,
      "snippet": "9e02b3"
     }
    ]
   },
   {
    "summary": "An approved contractor registry excerpt mapping contractor codes to names and maintenance scopes. It may help identify contractors in site paperwork: CT-118 is Halloran Services Ltd (mechanical and electrical maintenance), and CT-204 is Merrow Facilities Group (HVAC and compressed air). It provides no Q3 equipment-incident counts, so it cannot establish which contractor had the most incidents.",
    "provenance": [],
    "quotes": [
     {
      "text": "CT-118 .... Halloran Services Ltd — mechanical and electrical maintenance",
      "start": 54,
      "end": 127,
      "snippet": "d9add2"
     },
     {
      "text": "CT-204 .... Merrow Facilities Group — HVAC and compressed air",
      "start": 128,
      "end": 189,
      "snippet": "d9add2"
     },
     {
      "text": "Codes are used in site paperwork; the registry is the authority for the mapping.",
      "start": 248,
      "end": 328,
      "snippet": "d9add2"
     }
    ]
   },
   {
    "summary": "",
    "provenance": [],
    "quotes": []
   }
  ],
  [
   {
    "summary": "A Q3 operations memo for the Northgate site, reporting 5 equipment incidents under contractor code CT-118. The registry identifies CT-118 as Halloran Services Ltd; these incidents add to that contractor’s overall tally.",
    "provenance": [
     2,
     3
    ],
    "quotes": [
     {
      "text": "Equipment incidents logged this quarter: 5.",
      "start": 45,
      "end": 88,
      "snippet": "17f027"
     },
     {
      "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
      "start": 141,
      "end": 218,
      "snippet": "17f027"
     }
    ]
   },
   {
    "summary": "A Q3 operations-review memo for the Eastfield site. It reports 9 equipment incidents and identifies contractor code CT-204 for planned maintenance; the contractor registry maps CT-204 to Merrow Facilities Group. This adds 9 incidents to that contractor’s total.",
    "provenance": [
     3
    ],
    "quotes": [
     {
      "text": "Equipment incidents logged this quarter: 9.",
      "start": 45,
      "end": 88,
      "snippet": "964bc2"
     },
     {
      "text": "Planned maintenance at this site is carried out under contractor code CT-204.",
      "start": 136,
      "end": 213,
      "snippet": "964bc2"
     }
    ]
   },
   {
    "summary": "A Q3 operations-review memo for the Riverside site. It reports 6 equipment incidents under maintenance contractor code CT-118, identified in the contractor registry as Halloran Services Ltd; this is Riverside’s contribution to the contractor comparison.",
    "provenance": [
     3
    ],
    "quotes": [
     {
      "text": "Equipment incidents logged this quarter: 6.",
      "start": 45,
      "end": 88,
      "snippet": "9e02b3"
     },
     {
      "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
      "start": 164,
      "end": 241,
      "snippet": "9e02b3"
     }
    ]
   },
   {
    "summary": "An approved contractor-registry extract that maps site paperwork codes to contractor names. It identifies CT-118 as Halloran Services Ltd; the site reports show CT-118 sites logged 11 incidents in total, more than CT-204’s 9, so this document supplies the contractor name for the leading total.",
    "provenance": [
     0,
     1,
     2
    ],
    "quotes": [
     {
      "text": "CT-118 .... Halloran Services Ltd — mechanical and electrical maintenance",
      "start": 54,
      "end": 127,
      "snippet": "d9add2"
     },
     {
      "text": "Codes are used in site paperwork; the registry is the authority for the mapping.",
      "start": 248,
      "end": 328,
      "snippet": "d9add2"
     }
    ]
   },
   {
    "summary": "",
    "provenance": [],
    "quotes": []
   }
  ]
 ],
 "schema": {
  "code": "from pydantic import BaseModel, Field\n\nclass EquipmentIncidentReport(BaseModel):\n    \"\"\"A site's reported equipment-incident count associated with a contractor.\"\"\"\n    site: str | None = Field(description=\"The site named in the report\")\n    period: str | None = Field(description=\"The reporting period\")\n    incident_type: str | None = Field(description=\"The incident category as stated\")\n    incident_count: int | None = Field(description=\"The number of incidents reported\")\n    contractor_code: str | None = Field(description=\"The contractor code associated with the incidents\")\n\nclass ContractorMapping(BaseModel):\n    \"\"\"A mapping from a paperwork code to a contractor name.\"\"\"\n    contractor_code: str | None = Field(description=\"The code shown in the registry\")\n    contractor_name: str | None = Field(description=\"The contractor name mapped to that code\")\n\nclass Parse(BaseModel):\n    equipment_incident_reports: list[EquipmentIncidentReport]\n    contractor_mappings: list[ContractorMapping]",
  "thought": "The comparison must be computed across site-level reports, then linked to contractor names using registry mappings. Capture each report’s site, period, incident type, count, and contractor code, plus each code-to-name mapping; do not precompute totals."
 },
 "parsed": [
  {
   "parse": {
    "equipment_incident_reports": [
     {
      "site": "Northgate",
      "period": "Q3",
      "incident_type": "equipment incidents",
      "incident_count": 5,
      "contractor_code": "CT-118"
     }
    ],
    "contractor_mappings": []
   },
   "provenance": []
  },
  {
   "parse": {
    "equipment_incident_reports": [
     {
      "site": "Eastfield",
      "period": "Q3",
      "incident_type": "equipment incidents",
      "incident_count": 9,
      "contractor_code": "CT-204"
     }
    ],
    "contractor_mappings": []
   },
   "provenance": []
  },
  {
   "parse": {
    "equipment_incident_reports": [
     {
      "site": "Riverside",
      "period": "Q3",
      "incident_type": "equipment incidents",
      "incident_count": 6,
      "contractor_code": "CT-118"
     }
    ],
    "contractor_mappings": []
   },
   "provenance": []
  },
  {
   "parse": {
    "equipment_incident_reports": [],
    "contractor_mappings": [
     {
      "contractor_code": "CT-118",
      "contractor_name": "Halloran Services Ltd"
     },
     {
      "contractor_code": "CT-204",
      "contractor_name": "Merrow Facilities Group"
     },
     {
      "contractor_code": "CT-311",
      "contractor_name": "Bellweather Grounds"
     }
    ]
   },
   "provenance": []
  },
  {
   "parse": {
    "equipment_incident_reports": [],
    "contractor_mappings": []
   },
   "provenance": []
  }
 ],
 "answer": {
  "text": "Halloran Services Ltd’s sites logged the most equipment incidents in Q3, with 11 in total (5 at Northgate and 6 at Riverside).",
  "provenance": {
   "summary_ids": [
    "17f027",
    "9e02b3",
    "964bc2",
    "d9add2"
   ],
   "parse_ids": [
    "17f027",
    "9e02b3",
    "964bc2",
    "d9add2"
   ],
   "quotes": [
    {
     "document": "17f027",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 5.",
     "name": "0"
    },
    {
     "document": "17f027",
     "start": 141,
     "end": 218,
     "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
     "name": "0"
    },
    {
     "document": "964bc2",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 9.",
     "name": "1"
    },
    {
     "document": "964bc2",
     "start": 136,
     "end": 213,
     "text": "Planned maintenance at this site is carried out under contractor code CT-204.",
     "name": "1"
    },
    {
     "document": "9e02b3",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 6.",
     "name": "2"
    },
    {
     "document": "9e02b3",
     "start": 164,
     "end": 241,
     "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
     "name": "2"
    },
    {
     "document": "d9add2",
     "start": 54,
     "end": 127,
     "text": "CT-118 .... Halloran Services Ltd — mechanical and electrical maintenance",
     "name": "3"
    },
    {
     "document": "d9add2",
     "start": 128,
     "end": 189,
     "text": "CT-204 .... Merrow Facilities Group — HVAC and compressed air",
     "name": "3"
    },
    {
     "document": "d9add2",
     "start": 248,
     "end": 328,
     "text": "Codes are used in site paperwork; the registry is the authority for the mapping.",
     "name": "3"
    }
   ],
   "trail": {
    "rule": "through-summaries",
    "nodes": [
     "answer",
     "summary:2:17f027",
     "summary:2:9e02b3",
     "summary:2:964bc2",
     "summary:2:d9add2",
     "parse:17f027",
     "parse:9e02b3",
     "parse:964bc2",
     "parse:d9add2",
     "summary:1:9e02b3",
     "summary:1:d9add2",
     "document:17f027",
     "document:9e02b3",
     "document:964bc2",
     "summary:1:17f027",
     "summary:1:964bc2",
     "document:d9add2"
    ],
    "edges": [
     [
      "answer",
      "summary:2:17f027",
      "cites"
     ],
     [
      "answer",
      "summary:2:9e02b3",
      "cites"
     ],
     [
      "answer",
      "summary:2:964bc2",
      "cites"
     ],
     [
      "answer",
      "summary:2:d9add2",
      "cites"
     ],
     [
      "answer",
      "parse:17f027",
      "cites"
     ],
     [
      "answer",
      "parse:9e02b3",
      "cites"
     ],
     [
      "answer",
      "parse:964bc2",
      "cites"
     ],
     [
      "answer",
      "parse:d9add2",
      "cites"
     ],
     [
      "summary:2:17f027",
      "summary:1:9e02b3",
      "cites"
     ],
     [
      "summary:2:17f027",
      "summary:1:d9add2",
      "cites"
     ],
     [
      "summary:2:17f027",
      "document:17f027",
      "summarizes"
     ],
     [
      "summary:2:9e02b3",
      "summary:1:d9add2",
      "cites"
     ],
     [
      "summary:2:9e02b3",
      "document:9e02b3",
      "summarizes"
     ],
     [
      "summary:2:964bc2",
      "summary:1:d9add2",
      "cites"
     ],
     [
      "summary:2:964bc2",
      "document:964bc2",
      "summarizes"
     ],
     [
      "summary:2:d9add2",
      "summary:1:17f027",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "summary:1:964bc2",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "summary:1:9e02b3",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "document:d9add2",
      "summarizes"
     ],
     [
      "summary:1:9e02b3",
      "document:9e02b3",
      "summarizes"
     ],
     [
      "summary:1:d9add2",
      "document:d9add2",
      "summarizes"
     ],
     [
      "summary:1:17f027",
      "document:17f027",
      "summarizes"
     ],
     [
      "summary:1:964bc2",
      "document:964bc2",
      "summarizes"
     ]
    ]
   }
  }
 },
 "calls": {
  "r1-4": {
   "step": 1,
   "tokens": 1011,
   "usd": 0.0001263
  },
  "r1-2": {
   "step": 2,
   "tokens": 1077,
   "usd": 0.00015729999999999997
  },
  "r1-0": {
   "step": 3,
   "tokens": 1071,
   "usd": 0.00015549999999999999
  },
  "r1-1": {
   "step": 4,
   "tokens": 1074,
   "usd": 0.00015739999999999998
  },
  "r1-3": {
   "step": 5,
   "tokens": 1158,
   "usd": 0.00019259999999999997
  },
  "r2-4": {
   "step": 6,
   "tokens": 1508,
   "usd": 0.0002128
  },
  "r2-1": {
   "step": 7,
   "tokens": 1657,
   "usd": 0.00030605
  },
  "r2-3": {
   "step": 8,
   "tokens": 1668,
   "usd": 0.000319425
  },
  "r2-2": {
   "step": 9,
   "tokens": 1752,
   "usd": 0.0003543
  },
  "r2-0": {
   "step": 10,
   "tokens": 1740,
   "usd": 0.0003498
  },
  "parsed-4": {
   "step": 1,
   "tokens": 2305,
   "usd": 5.3335e-05
  },
  "parsed-3": {
   "step": 2,
   "tokens": 2512,
   "usd": 0.000402425
  },
  "parsed-1": {
   "step": 3,
   "tokens": 2496,
   "usd": 0.000148835
  },
  "parsed-0": {
   "step": 4,
   "tokens": 2656,
   "usd": 0.00022846
  },
  "parsed-2": {
   "step": 5,
   "tokens": 2987,
   "usd": 0.000392085
  },
  "schema": {
   "step": 1,
   "tokens": 2382,
   "usd": 0.0004960499999999999
  },
  "answer": {
   "step": 1,
   "tokens": 3346,
   "usd": 0.00047705
  }
 },
 "totals": {
  "tokens": 32400,
  "usd": 0.004529715
 }
};

  // Real r3con.ask answers over SAMPLE's run folder.
  /** @type {RecordedAsk[]} */
  const RECORDED_ASKS = [
 {
  "question": "Which sites does Halloran Services maintain?",
  "model": "openai/gpt-6-luna",
  "folder": "20261105T173751Z_fffbcd10",
  "created": "2026-11-05T17:37:51+00:00",
  "text": "Halloran Services Ltd maintains the Northgate and Riverside sites.",
  "provenance": {
   "summary_ids": [
    "17f027",
    "9e02b3",
    "d9add2"
   ],
   "parse_ids": [
    "17f027",
    "9e02b3",
    "d9add2"
   ],
   "quotes": [
    {
     "document": "17f027",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 5.",
     "name": "0"
    },
    {
     "document": "17f027",
     "start": 141,
     "end": 218,
     "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
     "name": "0"
    },
    {
     "document": "964bc2",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 9.",
     "name": "1"
    },
    {
     "document": "964bc2",
     "start": 136,
     "end": 213,
     "text": "Planned maintenance at this site is carried out under contractor code CT-204.",
     "name": "1"
    },
    {
     "document": "9e02b3",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 6.",
     "name": "2"
    },
    {
     "document": "9e02b3",
     "start": 164,
     "end": 241,
     "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
     "name": "2"
    },
    {
     "document": "d9add2",
     "start": 54,
     "end": 127,
     "text": "CT-118 .... Halloran Services Ltd — mechanical and electrical maintenance",
     "name": "3"
    },
    {
     "document": "d9add2",
     "start": 128,
     "end": 189,
     "text": "CT-204 .... Merrow Facilities Group — HVAC and compressed air",
     "name": "3"
    },
    {
     "document": "d9add2",
     "start": 248,
     "end": 328,
     "text": "Codes are used in site paperwork; the registry is the authority for the mapping.",
     "name": "3"
    }
   ],
   "trail": {
    "rule": "through-summaries",
    "nodes": [
     "answer",
     "summary:2:17f027",
     "summary:2:9e02b3",
     "summary:2:d9add2",
     "parse:17f027",
     "parse:9e02b3",
     "parse:d9add2",
     "summary:1:9e02b3",
     "summary:1:d9add2",
     "document:17f027",
     "document:9e02b3",
     "summary:1:17f027",
     "summary:1:964bc2",
     "document:d9add2",
     "document:964bc2"
    ],
    "edges": [
     [
      "answer",
      "summary:2:17f027",
      "cites"
     ],
     [
      "answer",
      "summary:2:9e02b3",
      "cites"
     ],
     [
      "answer",
      "summary:2:d9add2",
      "cites"
     ],
     [
      "answer",
      "parse:17f027",
      "cites"
     ],
     [
      "answer",
      "parse:9e02b3",
      "cites"
     ],
     [
      "answer",
      "parse:d9add2",
      "cites"
     ],
     [
      "summary:2:17f027",
      "summary:1:9e02b3",
      "cites"
     ],
     [
      "summary:2:17f027",
      "summary:1:d9add2",
      "cites"
     ],
     [
      "summary:2:17f027",
      "document:17f027",
      "summarizes"
     ],
     [
      "summary:2:9e02b3",
      "summary:1:d9add2",
      "cites"
     ],
     [
      "summary:2:9e02b3",
      "document:9e02b3",
      "summarizes"
     ],
     [
      "summary:2:d9add2",
      "summary:1:17f027",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "summary:1:964bc2",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "summary:1:9e02b3",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "document:d9add2",
      "summarizes"
     ],
     [
      "summary:1:9e02b3",
      "document:9e02b3",
      "summarizes"
     ],
     [
      "summary:1:d9add2",
      "document:d9add2",
      "summarizes"
     ],
     [
      "summary:1:17f027",
      "document:17f027",
      "summarizes"
     ],
     [
      "summary:1:964bc2",
      "document:964bc2",
      "summarizes"
     ]
    ]
   }
  },
  "totals": {
   "tokens": 3313,
   "usd": 0.00010234
  }
 },
 {
  "question": "Which sites does Halloran Services maintain?",
  "model": "openai/gpt-6.1-terra",
  "folder": "20261105T173754Z_6af5ac2f",
  "created": "2026-11-05T17:37:54+00:00",
  "text": "Halloran Services Ltd maintains the Northgate and Riverside sites.",
  "provenance": {
   "summary_ids": [
    "17f027",
    "9e02b3",
    "d9add2"
   ],
   "parse_ids": [
    "17f027",
    "9e02b3",
    "d9add2"
   ],
   "quotes": [
    {
     "document": "17f027",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 5.",
     "name": "0"
    },
    {
     "document": "17f027",
     "start": 141,
     "end": 218,
     "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
     "name": "0"
    },
    {
     "document": "964bc2",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 9.",
     "name": "1"
    },
    {
     "document": "964bc2",
     "start": 136,
     "end": 213,
     "text": "Planned maintenance at this site is carried out under contractor code CT-204.",
     "name": "1"
    },
    {
     "document": "9e02b3",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 6.",
     "name": "2"
    },
    {
     "document": "9e02b3",
     "start": 164,
     "end": 241,
     "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
     "name": "2"
    },
    {
     "document": "d9add2",
     "start": 54,
     "end": 127,
     "text": "CT-118 .... Halloran Services Ltd — mechanical and electrical maintenance",
     "name": "3"
    },
    {
     "document": "d9add2",
     "start": 128,
     "end": 189,
     "text": "CT-204 .... Merrow Facilities Group — HVAC and compressed air",
     "name": "3"
    },
    {
     "document": "d9add2",
     "start": 248,
     "end": 328,
     "text": "Codes are used in site paperwork; the registry is the authority for the mapping.",
     "name": "3"
    }
   ],
   "trail": {
    "rule": "through-summaries",
    "nodes": [
     "answer",
     "summary:2:17f027",
     "summary:2:9e02b3",
     "summary:2:d9add2",
     "parse:17f027",
     "parse:9e02b3",
     "parse:d9add2",
     "summary:1:9e02b3",
     "summary:1:d9add2",
     "document:17f027",
     "document:9e02b3",
     "summary:1:17f027",
     "summary:1:964bc2",
     "document:d9add2",
     "document:964bc2"
    ],
    "edges": [
     [
      "answer",
      "summary:2:17f027",
      "cites"
     ],
     [
      "answer",
      "summary:2:9e02b3",
      "cites"
     ],
     [
      "answer",
      "summary:2:d9add2",
      "cites"
     ],
     [
      "answer",
      "parse:17f027",
      "cites"
     ],
     [
      "answer",
      "parse:9e02b3",
      "cites"
     ],
     [
      "answer",
      "parse:d9add2",
      "cites"
     ],
     [
      "summary:2:17f027",
      "summary:1:9e02b3",
      "cites"
     ],
     [
      "summary:2:17f027",
      "summary:1:d9add2",
      "cites"
     ],
     [
      "summary:2:17f027",
      "document:17f027",
      "summarizes"
     ],
     [
      "summary:2:9e02b3",
      "summary:1:d9add2",
      "cites"
     ],
     [
      "summary:2:9e02b3",
      "document:9e02b3",
      "summarizes"
     ],
     [
      "summary:2:d9add2",
      "summary:1:17f027",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "summary:1:964bc2",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "summary:1:9e02b3",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "document:d9add2",
      "summarizes"
     ],
     [
      "summary:1:9e02b3",
      "document:9e02b3",
      "summarizes"
     ],
     [
      "summary:1:d9add2",
      "document:d9add2",
      "summarizes"
     ],
     [
      "summary:1:17f027",
      "document:17f027",
      "summarizes"
     ],
     [
      "summary:1:964bc2",
      "document:964bc2",
      "summarizes"
     ]
    ]
   }
  },
  "totals": {
   "tokens": 3288,
   "usd": 0.004525500000000001
  }
 },
 {
  "question": "How many incidents did Merrow Facilities Group's sites log?",
  "model": "openai/gpt-6-luna",
  "folder": "20261105T173756Z_3d3170c2",
  "created": "2026-11-05T17:37:56+00:00",
  "text": "Merrow Facilities Group's site(s) logged 9 equipment incidents in Q3.",
  "provenance": {
   "summary_ids": [
    "964bc2",
    "d9add2"
   ],
   "parse_ids": [
    "964bc2",
    "d9add2"
   ],
   "quotes": [
    {
     "document": "17f027",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 5.",
     "name": "0"
    },
    {
     "document": "17f027",
     "start": 141,
     "end": 218,
     "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
     "name": "0"
    },
    {
     "document": "964bc2",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 9.",
     "name": "1"
    },
    {
     "document": "964bc2",
     "start": 136,
     "end": 213,
     "text": "Planned maintenance at this site is carried out under contractor code CT-204.",
     "name": "1"
    },
    {
     "document": "9e02b3",
     "start": 45,
     "end": 88,
     "text": "Equipment incidents logged this quarter: 6.",
     "name": "2"
    },
    {
     "document": "9e02b3",
     "start": 164,
     "end": 241,
     "text": "Planned maintenance at this site is carried out under contractor code CT-118.",
     "name": "2"
    },
    {
     "document": "d9add2",
     "start": 54,
     "end": 127,
     "text": "CT-118 .... Halloran Services Ltd — mechanical and electrical maintenance",
     "name": "3"
    },
    {
     "document": "d9add2",
     "start": 128,
     "end": 189,
     "text": "CT-204 .... Merrow Facilities Group — HVAC and compressed air",
     "name": "3"
    },
    {
     "document": "d9add2",
     "start": 248,
     "end": 328,
     "text": "Codes are used in site paperwork; the registry is the authority for the mapping.",
     "name": "3"
    }
   ],
   "trail": {
    "rule": "through-summaries",
    "nodes": [
     "answer",
     "summary:2:964bc2",
     "summary:2:d9add2",
     "parse:964bc2",
     "parse:d9add2",
     "summary:1:d9add2",
     "document:964bc2",
     "summary:1:17f027",
     "summary:1:964bc2",
     "summary:1:9e02b3",
     "document:d9add2",
     "document:17f027",
     "document:9e02b3"
    ],
    "edges": [
     [
      "answer",
      "summary:2:964bc2",
      "cites"
     ],
     [
      "answer",
      "summary:2:d9add2",
      "cites"
     ],
     [
      "answer",
      "parse:964bc2",
      "cites"
     ],
     [
      "answer",
      "parse:d9add2",
      "cites"
     ],
     [
      "summary:2:964bc2",
      "summary:1:d9add2",
      "cites"
     ],
     [
      "summary:2:964bc2",
      "document:964bc2",
      "summarizes"
     ],
     [
      "summary:2:d9add2",
      "summary:1:17f027",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "summary:1:964bc2",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "summary:1:9e02b3",
      "cites"
     ],
     [
      "summary:2:d9add2",
      "document:d9add2",
      "summarizes"
     ],
     [
      "summary:1:d9add2",
      "document:d9add2",
      "summarizes"
     ],
     [
      "summary:1:17f027",
      "document:17f027",
      "summarizes"
     ],
     [
      "summary:1:964bc2",
      "document:964bc2",
      "summarizes"
     ],
     [
      "summary:1:9e02b3",
      "document:9e02b3",
      "summarizes"
     ]
    ]
   }
  },
  "totals": {
   "tokens": 3353,
   "usd": 0.00012084
  }
 }
];

  // Replay timing, in milliseconds. A stage's nodes appear one by one, then settle in the order the
  // recorded run finished them.
  const TIMING = {
    start: 300,
    minSettleGap: 150,
    maxStageSpread: 1600,
    phaseGap: 450,
    schemaWork: 1400,
    answerWork: 1700,
    litDelay: 650,
  };
  const PACE = {
    read: { appearStagger: 150, settleFirst: 450, settleStagger: 150, minWork: 350 },
    model: { appearStagger: 260, settleFirst: 900, settleStagger: 380, minWork: 500 },
  };
  const MARK = '❀';
  const SPIN_FRAMES = ['·', '✾', '✿', '❀', '❁', '❀', '✿', '✾'];
  const SPIN_FRAME_MS = 140;

  const GRID = { colGap: 230, rowGap: 112 };
  /** @param {string} name */
  const cssPx = (name) => parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
  const ICON_PX = cssPx('--icon-size');
  const DOC_ICON_PX = cssPx('--doc-icon-size');
  const ICON_HALF = ICON_PX / 2;
  const DOC_ICON_HALF = DOC_ICON_PX / 2;
  const ICON_STROKE = 1.7;
  // The document icon is drawn larger, so its stroke is thinned to the same on-screen weight.
  const DOC_ICON_STROKE = (ICON_STROKE * ICON_PX) / DOC_ICON_PX;
  const EDGE_GAP = 6;
  const MIN_CURVE = 40;
  const LABEL_LINE_PX = 21;
  // Above a row and below the labels of the row over it: where its bridges run, a lane per node they hop.
  const LANE_ROOM = GRID.rowGap - 2 * ICON_HALF - 2 * LABEL_LINE_PX;
  const BRIDGE_RAMP = 40;
  const BRIDGE_CLEARANCE = ICON_HALF + LANE_ROOM;
  const LABEL_MAX = 24;
  const NODE_PAD_PX = 12;
  const ZOOM = { min: 0.05, max: 2.5, fitMax: 1.15 };
  // About this many rows fit a laptop's panel readably, so a tab of more documents folds those with nothing relevant.
  const FOLD_FROM_DOCS = 10;
  const MIN_FOLD = 3;
  // Before a run, a fold keeps this many documents in view at each end, as a diff keeps lines of context.
  const CONTEXT_DOCS = 3;
  // An opened fold lower in the panel than this share of its height glides up, so that the rows it opened show.
  const UNFOLD_TOP = 0.4;
  // A fold's line keeps the strongest kind of the lines it stands for.
  const EDGE_RANK = { context: 0, self: 1, flow: 2, cite: 3 };
  const FIT_MARGIN = 28;
  const WHEEL_ZOOM_RATE = 0.004;
  const WHEEL_LINE_PX = 16;
  const DRAG_THRESHOLD = 4;
  const POPOVER_GAP = 12;
  const POPOVER_MARGIN = 10;
  const JUMP_MARGIN = 40;
  const GLIDE_MS = 350;

  const SUMMARY_CHARS = 140;
  const EXCERPT_CHARS = 60;
  const PREVIEW_CHARS = 20000;
  const PREVIEW_LINES = 5000;
  const JSON_INDENT = 2;
  const PDF_SUFFIX = '.pdf';
  // Office's lock file for a document someone has open: ~$report.docx.
  const LOCK_FILE_PREFIX = '~$';
  // Where macOS's Finder puts each file's metadata in a zip it makes.
  const MACOS_METADATA_FOLDER = '__MACOSX';
  // What r3con app extracts the text of: r3con's own list, which tests/app/test_server.py holds this to.
  const EXTRACTED_SUFFIXES = ['.pdf', '.docx'];
  const PLAIN_TEXT_SUFFIXES = ['.txt', '.md', '.markdown', '.rst', '.json', '.jsonl', '.csv', '.tsv', '.yaml', '.yml', '.html', '.xml'];
  // What r3con's folder walk reads, its TEXT_SUFFIXES, which tests/app/test_server.py holds this to.
  const FOLDER_SUFFIXES = [...PLAIN_TEXT_SUFFIXES, ...EXTRACTED_SUFFIXES];
  const ZIP_SUFFIX = '.zip';
  // The browser keeps about six connections to r3con app: a batch's extractions leave the rest to its other requests.
  const EXTRACTIONS_AT_ONCE = 4;
  const OLD_WORD_SUFFIX = '.doc';
  // A NUL byte this near a file's start means it isn't text, as git judges it.
  const SNIFF_BYTES = 8000;
  const UTF16_BOMS = [
    [0xff, 0xfe],
    [0xfe, 0xff],
  ];
  const MAX_SIMULATED_CITES = 4;
  const SIMULATED_SUMMARY_STRIDE = 2;
  const SIMULATED_PARSE_STRIDE = 3;
  // A made-up answer's trail goes one hop, under none of r3con's rules.
  const ONE_HOP = 'one hop';
  const SIMULATED_LABEL = 'simulated — not connected to R3Con yet';
  const SIMULATED_SCHEMA = [
    '# simulated — not connected to R3Con yet',
    'from pydantic import BaseModel',
    '',
    'class Excerpt(BaseModel):',
    '    source: str',
    '    excerpt: str',
    '',
    'class Parse(BaseModel):',
    '    excerpts: list[Excerpt]',
  ].join('\n');
  const THEME_KEY = 'r3con-proto-theme';
  const MODEL_KEY = 'r3con-proto-model';
  const ROUNDS_KEY = 'r3con-proto-rounds';
  const SIDE_WIDTH_KEY = 'r3con-proto-side-width';
  // The side column's narrowest, where the composer and the files pane still fit, its widest as a share of the frame, and a key press's step.
  const SIDE_MIN_PX = 280;
  const SIDE_MAX_SHARE = 0.6;
  const SIDE_STEP_PX = 16;
  const SAVED_MODELS_KEY = 'r3con-proto-saved-models';
  const SAVED_GROUP = 'saved';
  const SERVER_LABEL = 'Server';
  // r3con app's own words for a server URL it refuses.
  const CREDENTIALS_IN_URL = 'must not hold a user or password: put the key in its key field';
  const NO_HOST_IN_URL = 'must name a host';
  // Earlier prototypes kept a model and an endpoint here.
  const RETIRED_SETTINGS_KEY = 'r3con-proto-settings';
  const DEFAULT_MODEL = 'openai/gpt-6-luna';
  // Models taken from LiteLLM's model list on 2026-10-29, and updated by hand.
  /** @type {Record<string, { label: string, models: string[] }>} */
  const PROVIDERS = {
    openai: { label: 'OpenAI', models: ['openai/gpt-6.1-sol', 'openai/gpt-6.1-terra', 'openai/gpt-6-luna'] },
    anthropic: { label: 'Anthropic', models: ['anthropic/claude-fable-5-1', 'anthropic/claude-opus-5-5', 'anthropic/claude-sonnet-5-5'] },
    gemini: { label: 'Google', models: ['gemini/gemini-3.8-flash', 'gemini/gemini-3.1-pro-preview'] },
  };
  const MASK = '••••';
  // r3con app's own hint rule, which the prototype follows: a key's last few characters, if they are a small part of it.
  const KEY_HINT_CHARS = 4;
  const MIN_HINTED_KEY_CHARS = 3 * KEY_HINT_CHARS;
  const SAVED_MS = 1800;
  const ARRANGE_MS = 420;
  const FAINT_FADE_FROM_DOCS = 5;
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const ARIA_KIND = { doc: 'document', parsed: 'parsed structure', schema: 'proposed schema' };
  const STAGE_PHASE = { schema: 'proposing a structure', parsed: 'parsing', answer: 'reasoning' };
  // The relevance rounds r3con's default config runs, and a new run's until another count is chosen in the settings.
  const DEFAULT_ROUNDS = 2;
  // The most r3con app takes, its MAX_RELEVANCE_ROUNDS.
  const MAX_ROUNDS = 4;
  const LOOPBACK = ['127.0.0.1', 'localhost'];
  const UNAUTHORIZED = 401;
  const STALE_TOKEN = "this page's launch token is missing or stale; reopen it from the r3con app terminal";
  const LOST_CONNECTION = 'lost the connection to r3con app';
  const NOT_TEXT = 'R3Con reads text, PDF and Word (.docx) files: save as one of those, then add again.';
  const OLD_WORD = "R3Con reads Word's .docx, not the older .doc: save as .docx, PDF or text, then add again.";
  const ASK_WHEN_DONE = 'this run is still going; ask once it has finished';
  /** @param {Tab} tab */
  const tabStillGoing = (tab) => `tab ${tab.id} is still going; close it once it has finished`;
  const TYPE_A_MODEL = 'type a model first';
  const ENDINGS = new Set(['run_finished', 'run_failed', 'ask_finished', 'ask_failed']);
  const STOPPING = 'stopping · the call in flight finishes first';
  // The r3con stage, as its events name it, that each kind of node but a round's is drawn in; a replay reads first.
  const STAGE_OF = { schema: 'schema', parsed: 'parsing', answer: 'reasoning' };
  const READING = 'reading';
  const PROTOTYPE_KEYS_NOTE = 'This prototype sends nothing.';
  const PROTOTYPE_MODE_NOTE = 'prototype · replays one recorded run';
  const NEEDS_APP = 'PDF, Word and zip files need r3con app; this prototype reads only text files.';
  /** @param {number} skipped */
  const noDocuments = (skipped) => `No text, PDF or Word (.docx) files in it${skipped ? `; ${skipped} skipped` : ''}.`;
  const BROWSER_COULD_NOT_READ = "The browser couldn't read it.";
  // A refusal names this many files, then counts the rest.
  const NAMED_AT_MOST = 5;
  const NO_TEXT_LAYER = "No text layer: a scanned PDF needs OCR first, or it isn't a readable PDF.";
  const NO_TEXT = "No text: it's empty or only images, or R3Con can't read it.";
  const FILES_LOCKED = "These are this tab's files. Press + for a new tab.";
  const NO_FILES = 'no files yet';
  const ADD_FILES = `${NO_FILES} · add them in the files tab`;
  const REMOVE_ALL = 'remove all';
  const REMOVE_ALL_NAME = 'remove all documents';
  const UNDO = 'undo';
  /** @param {string} count */
  const undoName = (count) => `undo remove all (${count})`;
  // A press on remove all or undo this soon after the last is the same gesture going on: a double-click, a held key.
  const PRESS_GUARD_MS = 400;
  const LOCKED_DROP = 'This tab has run, so its files are locked: press + for a new tab.';
  const RAN_FIRST = "The tab's run started first, so its files are locked.";
  const DROP_ADDS = 'drop to add files';
  const DROP_LOCKED = 'this tab has run · press + for a new tab';

  const ICONS = {
    chevron: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M6 3.5 10.5 8 6 12.5"/>
    </svg>`,
    fold: `<svg viewBox="0 0 32 32" fill="currentColor" aria-hidden="true">
      <circle cx="16" cy="7.5" r="3"/><circle cx="16" cy="16" r="3"/><circle cx="16" cy="24.5" r="3"/>
    </svg>`,
    document: `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="${DOC_ICON_STROKE}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M19.25 3H8.5a2 2 0 0 0-2 2v22a2 2 0 0 0 2 2h15a2 2 0 0 0 2-2V9.25z"/>
      <path d="M19.25 3v5.25a1 1 0 0 0 1 1h5.25"/>
      <path d="M10.75 13h7.5M10.75 17.25h10.25M10.75 21.5h7.5"/>
    </svg>`,
    snippet: `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="${ICON_STROKE}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M15.5 4.75h2.3a2 2 0 0 1 1.46.63l5.17 5.55a2 2 0 0 1 .54 1.37V25.5a2.5 2.5 0 0 1-2.5 2.5H10a2.5 2.5 0 0 1-2.5-2.5v-9"/>
      <path d="M18.75 5.25V9.5a1.5 1.5 0 0 0 1.5 1.5h4.5"/>
      <path d="M11.75 16.25h9M11.75 19.5h9M15.75 22.75h5"/>
      <path d="M9 2.8c.45 3.15 1.25 3.95 4.4 4.4-3.15.45-3.95 1.25-4.4 4.4-.45-3.15-1.25-3.95-4.4-4.4 3.15-.45 3.95-1.25 4.4-4.4z" fill="currentColor" stroke-width="1"/>
    </svg>`,
    /** @param {boolean} filled */
    tree: (filled) => {
      /** @param {string} cls */
      const f = (cls) => (filled ? ` class="${cls}"` : '');
      return `<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" aria-hidden="true">
        <path d="M15.75 5v2.5M7.8 7.5h15.8M7.8 7.5v2M23.6 7.5v2.5M7.8 14.5v2.1M4 16.6h8M4 16.6v1.9M12 16.6v1.9M23.6 13.75v2.75M23.6 21.7v1.8M18.4 23.5h10.4M18.4 23.5v1.7M23.6 23.5v1.7M28.8 23.5v1.7"/>
        <g${f('ink')}>
        <rect x="10.5" y="1.5" width="10.5" height="3.5" rx=".4"${f('fill-yellow')}/>
        <circle cx="7.8" cy="12" r="2.5"${f('fill-orange')}/>
        <rect x="19.7" y="10" width="7.8" height="3.75" rx=".4"${f('fill-orange')}/>
        <rect x="1.2" y="18.5" width="5.6" height="3.4" rx=".4"${f('fill-blue')}/>
        <rect x="9.2" y="18.5" width="5.6" height="3.4" rx=".4"${f('fill-blue')}/>
        <circle cx="23.6" cy="19.1" r="2.6"${f('fill-yellow')}/>
        <circle cx="18.4" cy="27.5" r="2.3"${f('fill-blue')}/>
        <circle cx="23.6" cy="27.5" r="2.3"${f('fill-blue')}/>
        <circle cx="28.8" cy="27.5" r="2.3"${f('fill-blue')}/>
        </g>
      </svg>`;
    },
  };

  /**
   * @template {HTMLElement} [T=HTMLElement]
   * @param {string} id
   * @param {new () => T} [type]
   * @returns {T}
   */
  function $(id, type) {
    const el = document.getElementById(id);
    if (!(el instanceof (type ?? HTMLElement))) throw new Error(`the page has no #${id} of that type`);
    return /** @type {T} */ (el);
  }

  const root = document.documentElement;
  const panel = $('panel');
  const world = $('world');
  const inspector = $('inspector');
  const questionEl = $('question', HTMLTextAreaElement);
  const modelPick = $('model-pick', HTMLSelectElement);
  const modelOther = $('model-other', HTMLInputElement);
  const modelRemove = $('model-remove', HTMLButtonElement);
  const roundsPick = $('rounds', HTMLSelectElement);
  const frame = $('frame');
  const trash = $('trash');
  const removeAllButtons = [$('files-remove-all'), $('panel-remove-all')];
  const side = $('side');
  const resizer = $('resizer');

  // Served by r3con app on this computer the page runs r3con; anywhere else it is the prototype,
  // which sends nothing. The launch token lives only in this closure: it leaves the address bar at
  // once and never reaches storage.
  const live = ['http:', 'https:'].includes(location.protocol) && LOOPBACK.includes(location.hostname);
  const token = live ? takeToken() : null;

  function takeToken() {
    const found = new URLSearchParams(location.hash.slice(1)).get('token');
    if (found !== null) history.replaceState(null, '', `${location.pathname}${location.search}`);
    return found;
  }

  let nextUid = 0;
  let nextTab = 1;

  /** @param {Omit<LoadedFile, 'bytes' | 'uid'>} file @returns {LoadedFile} */
  function loaded(file) {
    return { ...file, bytes: new Blob([file.text]).size, uid: nextUid++ };
  }

  const SAMPLE_FILES = SAMPLE.docs.map((d) => loaded({ ...d, sample: true }));

  /** @param {LoadedFile} file */
  const docId = (file) => `doc-${file.uid}`;
  /** @param {NodeKind} kind */
  const iconHalf = (kind) => (kind === 'doc' ? DOC_ICON_HALF : ICON_HALF);
  // A node the caller knows the tab has drawn.
  /** @param {Tab} tab @param {string} id */
  const drawnNode = (tab, id) => /** @type {PanelNode} */ (tab.nodes.get(id));
  // A tab the caller knows has a run.
  /** @param {Tab} tab */
  const runTab = (tab) => /** @type {RunTab} */ (tab);

  // A tab's files are only ever reassigned, never changed in place, so tabs can share a list.
  /** @param {LoadedFile[]} files @param {View} [view] */
  function newTab(files, view = { x: 0, y: 0, k: 1 }) {
    const group = () => document.createElementNS(SVG_NS, 'g');
    const layers = { base: group(), mid: group(), lit: group() };
    const edges = document.createElementNS(SVG_NS, 'svg');
    edges.classList.add('edges');
    edges.setAttribute('aria-hidden', 'true');
    edges.append(layers.base, layers.mid, layers.lit);
    const nodesEl = h('div');
    const id = nextTab++;
    /** @type {Tab} */
    const tab = {
      id,
      files,
      removed: null,
      reading: 0,
      filesSaid: '',
      filesError: '',
      draft: '',
      adding: Promise.resolve(),
      button: h('button', { class: 'btn run-tab', id: `run-tab-${id}`, type: 'button', role: 'tab', 'aria-controls': 'panel', onclick: () => switchTab(tab) }, h('span', { class: 'num' }, `${id} `)),
      // The tab list holds only tabs, so × is for the mouse; the keyboard closes a tab with Delete.
      close: h('button', { class: 'btn tab-close', type: 'button', tabindex: '-1', 'aria-hidden': 'true', onclick: () => closeTab(tab) }, '×'),
      run: null,
      // A chat shown again isn't read out: a live region announces what changes in it, not what it holds.
      chat: h('div', { class: 'transcript', 'aria-live': 'polite', hidden: true }),
      chatToEnd: false,
      chatAnswer: null,
      turn: null,
      status: [],
      drawing: h('div', { class: 'drawing' }, edges, nodesEl),
      layers,
      nodesEl,
      nodes: new Map(),
      edges: new Map(),
      layout: new Map(),
      view: { ...view },
      userMoved: false,
      timers: [],
      narr: null,
      arrangeFrame: 0,
      trail: null,
      fold: null,
      placed: new Set(),
    };
    return tab;
  }

  /**
   * @type {{ tabs: Tab[], shown: Tab, streams: Set<Stream>,
   *   selected: string | null, hovered: string | null, inspectorDroppedAt: Offset | null }}
   */
  const state = {
    // Every tab, in order; init lists the first, the shown tab, with the sample memos.
    tabs: [],
    shown: newTab(SAMPLE_FILES),
    // The streams open on r3con app, a run's events or an ask's: each `{ kind, tab, job, stopping }`, with
    // the run or ask as `job`, dropped once closed.
    streams: new Set(),
    selected: null,
    hovered: null,
    inspectorDroppedAt: null,
  };

  // The recorded replay needs the whole sample set.
  /** @param {string} question @param {LoadedFile[]} [files] */
  const replaysRecording = (question, files = state.shown.files) => files === SAMPLE_FILES && question === SAMPLE.question;
  const fileCount = (files = state.shown.files) => plural(files.length, 'document');

  // ---------- small helpers ----------

  /**
   * @template {keyof HTMLElementTagNameMap} K
   * @param {K} tag
   * @param {Props} [props]
   * @param {...Child} children
   * @returns {HTMLElementTagNameMap[K]}
   */
  function h(tag, props = {}, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null || value === false) continue;
      if (key === 'class') el.className = String(value);
      else if (key === 'html') el.innerHTML = String(value);
      else if (key.startsWith('on')) el.addEventListener(key.slice(2), /** @type {EventListener} */ (value));
      else el.setAttribute(key, value === true ? '' : String(value));
    }
    el.append(...flatChildren(children));
    return el;
  }

  /** @param {Element} el @param {...Child} children */
  function fill(el, ...children) {
    el.replaceChildren(...flatChildren(children));
  }

  // TypeScript can't type what flat(Infinity) returns, hence the casts.
  /** @param {Child[]} children */
  function flatChildren(children) {
    const flat = /** @type {unknown[]} */ (children).flat(Infinity);
    return /** @type {(Node | string)[]} */ (flat.filter((c) => c != null && c !== false));
  }

  /** @param {number} v @param {number} lo @param {number} hi */
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  /** @param {number} n */
  const range = (n) => Array.from({ length: n }, (_, i) => i);
  /** @param {number} n */
  const fmtInt = (n) => n.toLocaleString('en-US');
  /** @param {number} n */
  const fmtUsd = (n) => `$${n < 0.01 ? n.toPrecision(2) : n.toFixed(2)}`;
  /** @param {number} n */
  const fmtBytes = (n) => (n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`);
  /** @param {number} n @param {string} one */
  function plural(n, one, many = `${one}s`) {
    return `${n} ${n === 1 ? one : many}`;
  }
  /** @param {string} name */
  const stripExt = (name) => name.replace(/\.[^./]+$/, '') || name;
  /** @param {string} s */
  const shortLabel = (s) => (s.length > LABEL_MAX ? `${s.slice(0, LABEL_MAX - 1)}…` : s);
  /** @param {LoadedFile} file */
  const fileLabel = (file) => shortLabel(stripExt(file.name));
  /** @param {{ name: string }} file @param {string} suffix */
  const hasSuffix = (file, suffix) => file.name.toLowerCase().endsWith(suffix);
  /** @param {string} model */
  const bareModel = (model) => model.split('/').pop();
  /** @param {LoadedFile[]} files */
  const totalBytes = (files) => files.reduce((sum, f) => sum + f.size, 0);
  /** @param {LoadedFile[]} files */
  const textBytes = (files) => files.reduce((sum, f) => sum + f.bytes, 0);
  const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Every event handled here starts at an element.
  /** @param {Event} event */
  const targetOf = (event) => /** @type {Element} */ (event.target);

  /** @param {string} key */
  function readStored(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  /** @param {string} key @param {string} value */
  function writeStored(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      // Storage can be blocked; the page works without it for this visit.
    }
  }

  /** @param {string} key */
  function removeStored(key) {
    try {
      localStorage.removeItem(key);
    } catch {
      // Blocked storage holds nothing to remove.
    }
  }

  /** @param {string} text @param {number} max */
  function excerpt(text, max) {
    const flat = text.replace(/\s+/g, ' ').trim();
    if (flat.length <= max) return flat;
    const cut = flat.slice(0, max);
    const space = cut.lastIndexOf(' ');
    return `${space > max * 0.6 ? cut.slice(0, space) : cut}…`;
  }

  /** @param {string} id */
  function splitId(id) {
    const [kind, doc] = id.split('-');
    return /** @type {[NodeKind, number]} */ ([kind, doc === undefined ? null : Number(doc)]);
  }

  /** @param {number} round @returns {RoundKind} */
  const roundKind = (round) => `r${round}`;
  /** @param {NodeKind} kind @returns {kind is RoundKind} */
  const isRound = (kind) => /^r\d+$/.test(kind);
  /** @param {RoundKind} kind */
  const roundOf = (kind) => Number(kind.slice(1));
  /** @param {RoundKind} kind */
  const roundName = (kind) => `round ${roundOf(kind)}`;
  /** @param {NodeKind} kind @returns {RoundKind | null} */
  const previousRound = (kind) => (isRound(kind) && roundOf(kind) > 1 ? roundKind(roundOf(kind) - 1) : null);
  /** @param {number} rounds @returns {Summary[][]} */
  const emptyRounds = (rounds) => range(rounds).map(() => []);
  /** @param {number} rounds */
  const roundKinds = (rounds) => range(rounds).map((k) => roundKind(k + 1));
  // The final round's summaries are what the structure, the parses and the answers are shown.
  /** @param {Pick<Run, 'rounds'>} run */
  const finalRound = (run) => roundKind(run.rounds.length);
  /** @param {Pick<Run, 'rounds'>} run @param {RoundKind} kind */
  const summariesIn = (run, kind) => run.rounds[roundOf(kind) - 1];
  /** @param {NodeKind} kind @returns {kind is PerDocumentKind} */
  const isPerDocument = (kind) => kind === 'parsed' || isRound(kind);
  // A kind of node's column in a panel of `rounds` rounds: the documents', each round's, the structure's, the
  // parses' and the answers'.
  /** @param {NodeKind} kind @param {number} rounds */
  const column = (kind, rounds) => (isRound(kind) ? roundOf(kind) : { doc: 0, schema: rounds + 1, parsed: rounds + 2, answer: rounds + 3 }[kind]);
  // The rounds a tab's panel has columns for: its run's, or r3con's default before it has one.
  /** @param {Tab} tab */
  const roundsOf = (tab) => tab.run?.rounds.length ?? DEFAULT_ROUNDS;

  /** @param {Run} run @param {number} i */
  function docLabel(run, i) {
    return stripExt(run.docs[i].name);
  }

  /** @param {Tab} tab @param {string} id */
  function nodeName(tab, id) {
    const [kind, d] = splitId(id);
    if (kind === 'doc') return tab.nodes.get(id)?.file?.name ?? 'document';
    if (kind === 'schema') return 'structure';
    const { run } = runTab(tab);
    if (kind === 'answer') return answerLabel(answerAt(run, id));
    return `${docLabel(run, d)} · ${kind === 'parsed' ? 'parsed' : roundName(kind)}`;
  }

  /** @param {string} code */
  function schemaClasses(code) {
    return [...code.matchAll(/class (\w+)\(BaseModel\)/g)].map((m) => m[1]).filter((c) => c !== 'Parse');
  }

  /** @param {Totals | null | undefined} totals */
  function spent(totals) {
    const { tokens = null, usd = null } = totals ?? {};
    return [tokens !== null && `${fmtInt(tokens)} tokens`, usd !== null && fmtUsd(usd)].filter(Boolean);
  }

  /** @param {Parse} parse */
  const hasRecords = (parse) => Object.values(parse.parse).some((records) => records.length > 0);

  // ---------- r3con app ----------

  // An error whose message is for the user: r3con app's own words, or what went wrong reaching it.
  class ServerError extends Error {}

  // A stale token or r3con app out of reach, kept for the status until a call to it succeeds.
  /** @type {string | null} */
  let connectionProblem = null;

  // What to tell the user about `error`; one that is not the server's is a bug, so it also goes to the console.
  /** @param {unknown} error @param {string} fallback */
  function explain(error, fallback, consoleText = fallback) {
    if (error instanceof ServerError) return error.message;
    console.error(consoleText, error);
    return fallback;
  }

  const lostConnection = () => new ServerError(LOST_CONNECTION);

  /** @param {string} path @param {{ method?: string, body?: unknown }} [request] */
  async function api(path, { method = 'GET', body } = {}) {
    const headers = { Authorization: `Bearer ${token}` };
    const json = body !== undefined && !(body instanceof Blob);
    let response;
    try {
      response = await fetch(path, {
        method,
        headers: json ? { ...headers, 'Content-Type': 'application/json' } : headers,
        body: json ? JSON.stringify(body) : /** @type {Blob | undefined} */ (body),
      });
    } catch {
      connectionProblem = LOST_CONNECTION;
      throw lostConnection();
    }
    if (response.status === UNAUTHORIZED) {
      connectionProblem = STALE_TOKEN;
      throw new ServerError(STALE_TOKEN);
    }
    if (response.ok) {
      connectionProblem = null;
      return response;
    }
    const reply = await response.json().catch(/** @returns {null} */ () => null);
    throw new ServerError(reply?.error ?? `r3con app answered ${response.status} ${response.statusText}`.trim());
  }

  /** @param {string} frame */
  function frameData(frame) {
    const fields = frame.split(/\r?\n/).filter((field) => field.startsWith('data:'));
    return fields.map((field) => field.slice('data:'.length).replace(/^ /, '')).join('\n');
  }

  // Server-sent events, read with fetch since EventSource can't send the token. A frame ends at a
  // blank line, and a chunk can end anywhere in one.
  /** @param {ReadableStream<BufferSource>} body @returns {AsyncGenerator<ServerEvent>} */
  async function* serverEvents(body) {
    const reader = body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    try {
      for (;;) {
        const { value, done } = await reader.read().catch(() => {
          throw lostConnection();
        });
        if (done) return;
        const frames = (buffer + value).split(/\r?\n\r?\n/);
        buffer = frames.pop() ?? '';
        for (const data of frames.map(frameData).filter(Boolean)) yield JSON.parse(data);
      }
    } finally {
      // Closes the response when drawing an event throws; a stream that already failed rejects with
      // the error the read threw.
      reader.cancel().catch(() => {});
    }
  }

  // Hands each of a run's or an ask's events to `onEvent`; a stream that stops before its last event lost the server.
  /** @param {string} path @param {(event: ServerEvent) => void} onEvent */
  async function follow(path, onEvent) {
    const response = await api(path);
    /** @type {ServerEvent | null} */
    let last = null;
    for await (const event of serverEvents(/** @type {ReadableStream<BufferSource>} */ (response.body))) {
      last = event;
      onEvent(event);
    }
    if (!last || !ENDINGS.has(last.type)) throw lostConnection();
  }

  /** @param {Tab} tab */
  const streamsOf = (tab) => [...state.streams].filter((stream) => stream.tab === tab);
  /** @param {Stream['kind']} kind @param {Tab} tab */
  const streaming = (kind, tab) => streamsOf(tab).some((stream) => stream.kind === kind);

  /** @param {Stream['kind']} kind @param {Tab} tab @param {Stream['job']} job @returns {Stream} */
  function openStream(kind, tab, job) {
    const stream = { kind, tab, job, stopping: false };
    state.streams.add(stream);
    return stream;
  }

  /** @param {Stream} stream */
  function closeStream(stream) {
    state.streams.delete(stream);
    if (stream.stopping) showStatus(stream.tab);
    updateComposer();
  }

  /** @param {Tab} tab */
  const stopping = (tab) => streamsOf(tab).some((stream) => stream.stopping);
  /** @param {Tab} tab */
  const toStop = (tab) => streamsOf(tab).filter((stream) => !stream.stopping);

  // A stop pressed before the job's id has come back is sent as soon as it does.
  /** @param {Stream} stream */
  function stopJob(stream) {
    stream.stopping = true;
    showStatus(stream.tab);
    if (stream.job.id) sendCancel(stream);
  }

  /** @param {Stream} stream @param {string} id */
  function jobStarted(stream, id) {
    stream.job.id = id;
    if (stream.stopping) sendCancel(stream);
  }

  // From the 202 on, r3con sends no new model call, and the stream ends with the job's last event.
  /** @param {Stream} stream */
  async function sendCancel(stream) {
    try {
      await api(`/api/${stream.kind}s/${stream.job.id}/cancel`, { method: 'POST' });
    } catch (error) {
      // A refusal that comes back once the job has ended has nothing left to stop.
      if (!state.streams.has(stream)) return;
      stream.stopping = false;
      updateComposer();
      if (stream.tab === state.shown) errorStatus(`couldn't stop · ${explain(error, /** @type {Error} */ (error).message, "couldn't stop")}`);
    }
  }

  /** @param {string[]} ids @returns {Pick<Run, 'ids' | 'indexes'>} */
  const withIds = (ids) => ({ ids, indexes: new Map(ids.map((id, d) => [id, d])) });

  // r3con names a snippet by its document's id, or `<id>.<k>` for part k of a document read in parts.
  /** @param {Pick<Run, 'indexes'>} run @param {string} id */
  const docIndex = (run, id) => run.indexes.get(id.split('.')[0]);

  /** @param {Pick<Run, 'indexes'>} run @param {string[]} ids */
  function docIndices(run, ids) {
    return [...new Set(ids.flatMap((id) => docIndex(run, id) ?? []))];
  }

  /** @param {Pick<Run, 'indexes'>} run @param {Omit<R3conProvenance, 'quotes'>} provenance @returns {Provenance} */
  function provenanceOf(run, { summary_ids, parse_ids, trail }) {
    return { summary_ids: docIndices(run, summary_ids), parse_ids: docIndices(run, parse_ids), trail };
  }

  /** @param {{ tokens: { total: number } | null, usd: number | null }} event */
  const totalsOf = (event) => ({ tokens: event.tokens?.total, usd: event.usd });
  /** @param {KeyStatus} status */
  const keyedProviders = (status) => Object.keys(PROVIDERS).filter((provider) => status[provider]);
  /** @param {KeyStatus} status */
  const serversOf = (status) => /** @type {Server[]} */ (status.other ?? []);
  /** @param {string} endpoint */
  const normalEndpoint = (endpoint) => endpoint.trim().replace(/\/+$/, '');
  // r3con app refuses these too, and the prototype must not keep what r3con app would refuse.
  /** @param {string} endpoint */
  function serverUrlRefusal(endpoint) {
    try {
      const url = new URL(endpoint);
      if (url.username || url.password) return CREDENTIALS_IN_URL;
      return url.hostname ? null : NO_HOST_IN_URL;
    } catch {
      return NO_HOST_IN_URL;
    }
  }

  // ---------- spinner ----------

  let spinFrame = 0;
  setInterval(() => {
    if (reduceMotion()) return;
    spinFrame = (spinFrame + 1) % SPIN_FRAMES.length;
    for (const el of document.querySelectorAll('.node.working .spin, .node.working .glyph, .log-spin')) el.textContent = SPIN_FRAMES[spinFrame];
  }, SPIN_FRAME_MS);

  // ---------- runs ----------

  /** @template {Quote} Q @param {Q} quote @returns {quote is Q & TextRange} */
  const located = (quote) => quote.start !== null;

  // r3con's names for the nodes of a trail.
  const TRAIL_ANSWER = 'answer';
  /** @param {number} round @param {string} snippet */
  const summaryNode = (round, snippet) => `summary:${round}:${snippet}`;
  /** @param {string} id */
  const parseNode = (id) => `parse:${id}`;
  /** @param {string} id */
  const documentNode = (id) => `document:${id}`;

  // Where each of `ranges` starts or ends, in order, once each.
  /** @param {{ start: number, end: number }[]} ranges */
  const cutsOf = (ranges) => [...new Set(ranges.flatMap((r) => [r.start, r.end]))].sort((a, b) => a - b);

  // `quotes`, found in `text`, with their places in UTF-16 code units, walking the text once up to the last of them.
  /** @template {Quote} Q @param {string} text @param {Q[]} quotes @returns {Q[]} */
  function quotesInUnits(text, quotes) {
    /** @type {Map<number, number>} */
    const units = new Map();
    let point = 0;
    let unit = 0;
    for (const target of cutsOf(quotes.filter(located))) {
      for (; point < target && unit < text.length; point += 1) unit += String.fromCodePoint(/** @type {number} */ (text.codePointAt(unit))).length;
      units.set(target, unit);
    }
    return quotes.map((q) => (located(q) ? /** @type {Q} */ ({ ...q, start: units.get(q.start), end: units.get(q.end) }) : q));
  }

  /** @returns {Run} */
  function sampleRun() {
    /** @param {RecordedSummary[]} summaries @param {number} round @returns {Summary[]} */
    const inUnits = (summaries, round) =>
      summaries.map((s, i) => ({
        ...s,
        quotes: quotesInUnits(SAMPLE.docs[i].text, s.quotes).map(({ snippet, ...q }) => ({ ...q, summary: summaryNode(round, snippet) })),
      }));
    /** @type {Omit<Run, 'answers'>} */
    const run = {
      simulated: false,
      docs: SAMPLE_FILES,
      ...withIds(SAMPLE.ids),
      rounds: SAMPLE.rounds.map((summaries, k) => inUnits(summaries, k + 1)),
      schema: SAMPLE.schema,
      parsed: SAMPLE.parsed,
      calls: SAMPLE.calls,
      totals: SAMPLE.totals,
      source: 'recorded run',
    };
    const { text, provenance } = SAMPLE.answer;
    return { ...run, answers: [{ question: SAMPLE.question, model: SAMPLE.model, text, provenance: provenanceOf(run, provenance) }] };
  }

  /** @param {number} i @param {number} n */
  function madeUpCitations(i, n) {
    if (n < 2) return [];
    const cites = [(i + 1) % n];
    if (i % 3 === 0 && n > 2) cites.push((i + 2) % n);
    return cites;
  }

  // As r3con does, it cites only non-empty final-round summaries and parses with records; its trail goes one hop.
  /** @param {Pick<Run, 'docs' | 'ids' | 'indexes' | 'rounds' | 'parsed'>} run @param {string} question @param {number} shift @returns {Pick<Answer, 'text' | 'provenance'>} */
  function madeUpAnswer(run, question, shift) {
    /** @param {number} stride @param {(i: number) => boolean} citable */
    const every = (stride, citable) =>
      range(run.docs.length).filter((i) => citable(i) && (i + shift) % stride === 0).slice(0, MAX_SIMULATED_CITES).map((d) => run.ids[d]);
    const final = summariesIn(run, finalRound(run));
    const summary_ids = every(SIMULATED_SUMMARY_STRIDE, (i) => Boolean(final[i].summary));
    const parse_ids = every(SIMULATED_PARSE_STRIDE, (i) => hasRecords(run.parsed[i]));
    /** @param {string} id */
    const summary = (id) => summaryNode(run.rounds.length, id);
    const cited = [...summary_ids.map(summary), ...parse_ids.map(parseNode)];
    const edges = [
      ...cited.map((to) => /** @type {TrailEdge} */ ([TRAIL_ANSWER, to, 'cites'])),
      ...summary_ids.map((id) => /** @type {TrailEdge} */ ([summary(id), documentNode(id), 'summarizes'])),
    ];
    const trail = { rule: ONE_HOP, nodes: [TRAIL_ANSWER, ...cited, ...summary_ids.map(documentNode)], edges };
    return {
      text: `R3Con's answer to “${question}” would appear here. The documents it cites are made up.`,
      provenance: provenanceOf(run, { summary_ids, parse_ids, trail }),
    };
  }

  /** @param {LoadedFile[]} files @param {string} question @param {string} model @param {number} rounds @returns {Run} */
  function simulatedRun(files, question, model, rounds) {
    const n = files.length;
    /** @param {LoadedFile} f */
    const summary = (f) => excerpt(f.text, SUMMARY_CHARS);
    const parses = files.map((f) => ({ excerpts: f.text.trim() ? [{ source: f.name, excerpt: excerpt(f.text, EXCERPT_CHARS) }] : [] }));
    // As in r3con: only a non-empty summary past round 1 or a parse with records cites, and only non-empty summaries.
    /** @param {number} j */
    const hasSummary = (j) => Boolean(summary(files[j]));
    /** @param {number} i @param {boolean} hasOutput */
    const madeUp = (i, hasOutput) => (hasOutput ? madeUpCitations(i, n).filter(hasSummary) : []);
    /** @param {number} k @returns {Summary[]} */
    const round = (k) => files.map((f, i) => ({ summary: summary(f), provenance: madeUp(i, k > 0 && hasSummary(i)), quotes: [] }));
    /** @type {Omit<Run, 'answers'>} */
    const run = {
      simulated: true,
      docs: files,
      ...withIds(range(n).map(String)),
      rounds: range(rounds).map(round),
      schema: { code: SIMULATED_SCHEMA, thought: `${SIMULATED_LABEL}. A real run proposes a schema for your question here.` },
      parsed: parses.map((parse, i) => ({ parse, provenance: madeUp(i, parse.excerpts.length > 0) })),
      calls: null,
      totals: null,
    };
    return { ...run, answers: [{ question, model, ...madeUpAnswer(run, question, 0) }] };
  }

  // Filled in by r3con app's events as they arrive; events carry no per-call tokens or cost, so it has no calls.
  // It is planned for the rounds it asks for, unless its run_started event says it has others.
  /** @param {string} question @param {string} model @param {string | null} endpoint @param {LoadedFile[]} docs @param {number} rounds @returns {Run} */
  function liveRun(question, model, endpoint, docs, rounds) {
    /** @type {Pick<Run, 'rounds' | 'schema' | 'parsed' | 'calls' | 'totals'>} */
    const empty = { rounds: emptyRounds(rounds), schema: null, parsed: [], calls: null, totals: null };
    return { simulated: false, id: null, ...withIds([]), docs, answers: [{ question, model, endpoint, text: null, provenance: null }], ...empty };
  }

  /** @param {boolean} simulated @param {Totals | null | undefined} totals */
  const spendOf = (simulated, totals) => (simulated ? ['nothing was sent'] : spent(totals));

  /** @param {Run} run */
  function runSummary(run) {
    const source = run.simulated ? 'simulated run' : run.source;
    return [source, run.answers[0].model, ...spendOf(run.simulated, run.totals)].filter(Boolean).join(' · ');
  }

  // An ask is `{ question, model, source, text, provenance, totals }`, its source recorded, simulated or live;
  // a live one also has the server's `id` for it.
  /** @param {Answer} ask */
  function askSummary({ source, model, totals }) {
    return [`${source} ask over this run`, model, ...spendOf(source === 'simulated', totals)].join(' · ');
  }

  // Answer k in run.answers is node `answer-k`; the run's own answer is the first.
  /** @param {number} k */
  const answerId = (k) => `answer-${k}`;
  const RUN_ANSWER = answerId(0);

  /** @param {Run} run @param {string} id */
  const answerAt = (run, id) => run.answers[splitId(id)[1]];

  // An answer with where it came from, as the chat and its popup give it.
  /** @param {Run} run @param {string} id */
  function answerOf(run, id) {
    const answer = answerAt(run, id);
    if (id === RUN_ANSWER) return { ...answer, simulated: run.simulated, summary: runSummary(run) };
    return { ...answer, simulated: answer.source === 'simulated', summary: askSummary(answer) };
  }

  /** @param {{ question: string, model: string }} answer */
  const answerLabel = ({ question, model }) => `${excerpt(question, LABEL_MAX)} · ${bareModel(model)}`;

  // An answer and a parse cite final-round summaries, and a summary of round k, those of round k - 1.
  /** @param {Run} run @param {string} id */
  function cites(run, id) {
    const [kind, i] = splitId(id);
    const final = finalRound(run);
    if (kind === 'answer') {
      // Only an answer that has landed is asked what it cites.
      const { summary_ids, parse_ids } = /** @type {Provenance} */ (answerAt(run, id).provenance);
      return [...summary_ids.map((d) => `${final}-${d}`), ...parse_ids.map((d) => `parsed-${d}`)];
    }
    if (kind === 'parsed') return run.parsed[i].provenance.map((d) => `${final}-${d}`);
    if (!isRound(kind)) return [];
    const previous = previousRound(kind);
    return previous ? summariesIn(run, kind)[i].provenance.map((d) => `${previous}-${d}`) : [];
  }

  /** @param {Run} run @param {string} id */
  function ownDocument(run, id) {
    const [kind, i] = splitId(id);
    return isPerDocument(kind) ? docId(run.docs[i]) : null;
  }

  // Trail node `node` as the page names it, or null where the page has no node for it.
  /** @param {Run} run @param {string} answerId @param {string} node */
  function pageNode(run, answerId, node) {
    if (node === TRAIL_ANSWER) return answerId;
    const [kind, ...rest] = node.split(':');
    const d = docIndex(run, rest[rest.length - 1]);
    if (d === undefined || !run.docs[d]) return null;
    if (kind === 'parse') return run.parsed[d] ? `parsed-${d}` : null;
    if (kind === 'document') return docId(run.docs[d]);
    const round = Number(rest[0]);
    return run.rounds[round - 1]?.[d] ? `${roundKind(round)}-${d}` : null;
  }

  // r3con walks the trail from the answer, so a summary's first edge on it comes from the node nearest the answer.
  /** @param {Run} run @param {string} answerId @returns {Trail | null} none until the answer lands, or if none of it is on the page */
  function trailOf(run, answerId) {
    const trail = answerAt(run, answerId).provenance?.trail;
    if (!trail) return null;
    const named = new Map(trail.nodes.map((node) => [node, pageNode(run, answerId, node)]));
    /** @type {Trail} */
    const drawn = { nodes: new Set(), edges: new Set(), citedBy: new Map(), quoting: new Set(), quoted: [] };
    for (const id of named.values()) if (id) drawn.nodes.add(id);
    for (const [from, to, kind] of trail.edges) {
      const [pageFrom, pageTo] = [named.get(from), named.get(to)];
      if (!pageFrom || !pageTo) continue;
      drawn.edges.add(`${pageTo}>${pageFrom}`);
      if (kind === 'cites' && to.startsWith('summary:') && !drawn.citedBy.has(pageTo)) drawn.citedBy.set(pageTo, pageFrom);
      if (kind === 'summarizes') drawn.quoting.add(from);
    }
    drawn.quoted = run.docs.map((_, d) => quotedIn(run, drawn.quoting, d));
    return drawn.edges.size ? drawn : null;
  }

  // The lit answer's trail is the tab's, unless its panel was cleared since.
  /** @param {RunTab} tab @param {string} id */
  const answerTrail = (tab, id) => (id === tab.run.lit && tab.trail) || trailOf(tab.run, id);

  // What `quoting` quotes of document d, by part and by sentence; quotes that overlap join, quotes that only meet don't.
  /** @param {Run} run @param {Set<string>} quoting the trail nodes of the summaries whose quotes count @param {number} d @returns {Quoted} */
  function quotedIn(run, quoting, d) {
    const quotes = run.rounds.flatMap((summaries, k) =>
      (summaries[d]?.quotes ?? []).filter((q) => quoting.has(q.summary)).filter(located).map((q) => ({ start: q.start, end: q.end, by: `${roundKind(k + 1)}-${d}` })),
    );
    const cuts = cutsOf(quotes);
    const spans = cuts.slice(1).map((end, i) => ({ start: cuts[i], end, by: [...new Set(quotes.filter((q) => q.start <= cuts[i] && q.end >= end).map((q) => q.by))] }));
    /** @type {TextRange[]} */
    const sentences = [];
    for (const { start, end } of quotes.toSorted((a, b) => a.start - b.start)) {
      const last = sentences.at(-1);
      if (last && start < last.end) last.end = Math.max(last.end, end);
      else sentences.push({ start, end });
    }
    return { spans: spans.filter((span) => span.by.length), sentences };
  }

  /** @param {Trail} trail */
  function sentencesFrom(trail) {
    const quoted = trail.quoted.filter(({ sentences }) => sentences.length);
    if (!quoted.length) return null;
    const sentences = quoted.reduce((sum, q) => sum + q.sentences.length, 0);
    return `${plural(sentences, 'sentence')} from ${plural(quoted.length, 'document')}`;
  }

  /** @param {Tab} tab @param {string} id */
  function citerName(tab, id) {
    const [kind, d] = splitId(id);
    if (kind === 'answer') return 'the answer';
    return kind === 'parsed' ? `the ${docLabel(runTab(tab).run, d)} parse` : nodeName(tab, id);
  }

  // A mark's hover text: each summary that quotes its span, and what on the trail cites that summary, if the page draws it.
  /** @param {Tab} tab @param {Trail} trail @param {Span} span */
  function quotedBy(tab, trail, span) {
    /** @param {string} id */
    const quoter = (id) => {
      const citer = trail.citedBy.get(id);
      return citer ? `${nodeName(tab, id)} (cited by ${citerName(tab, citer)})` : nodeName(tab, id);
    };
    return ['quoted by', ...span.by.map(quoter)].join('\n');
  }

  // ---------- panel: view ----------

  function applyView() {
    const { view } = state.shown;
    world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.k})`;
    world.style.setProperty('--k', String(view.k));
    $('zoom-level').textContent = `${Math.round(view.k * 100)}%`;
    positionInspector();
  }

  function glide() {
    if (reduceMotion()) return;
    world.classList.add('glide');
    setTimeout(() => world.classList.remove('glide'), GLIDE_MS);
  }

  /** @param {number} px @param {number} py @param {number} k */
  function zoomAt(px, py, k) {
    const { view } = state.shown;
    const next = clamp(k, ZOOM.min, ZOOM.max);
    const wx = (px - view.x) / view.k;
    const wy = (py - view.y) / view.k;
    view.k = next;
    view.x = px - wx * next;
    view.y = py - wy * next;
    applyView();
  }

  function measureCharWidth() {
    const probe = h('span', { class: 'probe' }, '0'.repeat(20));
    document.body.append(probe);
    const width = probe.getBoundingClientRect().width / 20;
    probe.remove();
    return width || 8;
  }

  const charPx = measureCharWidth();

  /** @param {LoadedFile[]} files @param {Origin} origin @param {number} rounds @returns {Map<string, Point>} */
  function gridLayout(files, origin, rounds) {
    const layout = new Map();
    const origins = files.map((f, i) => origin(i, f));
    if (!origins.length) return layout;
    /** @type {PerDocumentKind[]} */
    const perDocument = [...roundKinds(rounds), 'parsed'];
    files.forEach((f, i) => {
      layout.set(docId(f), origins[i]);
      for (const kind of perDocument) layout.set(`${kind}-${i}`, { x: origins[i].x + column(kind, rounds) * GRID.colGap, y: origins[i].y });
    });
    const ys = origins.map((o) => o.y);
    layout.set('schema', { x: origins[0].x + column('schema', rounds) * GRID.colGap, y: origins[0].y });
    layout.set(RUN_ANSWER, { x: Math.max(...origins.map((o) => o.x)) + column('answer', rounds) * GRID.colGap, y: (Math.min(...ys) + Math.max(...ys)) / 2 });
    return layout;
  }

  /** @param {number} i */
  const tidyOrigin = (i) => ({ x: 0, y: i * GRID.rowGap });

  /** @param {Tab} tab @returns {Origin} */
  function nodeOrigin(tab) {
    return (i, file) => {
      const node = drawnNode(tab, docId(file));
      return { x: node.x, y: node.y };
    };
  }

  // Answer k sits around the run's own answer, one row gap apart: below it, above it, then further out.
  /** @param {Point} first @param {number} k */
  function answerSpot(first, k) {
    const out = Math.ceil(k / 2);
    return { x: first.x, y: first.y + (k % 2 ? out : -out) * GRID.rowGap };
  }

  // The first spot around the run's answer, where it is now, that no answer node holds.
  /** @param {Tab} tab */
  function freeAnswerSpot(tab) {
    const first = drawnNode(tab, RUN_ANSWER);
    const answers = [...tab.nodes.values()].filter((n) => n.kind === 'answer');
    /** @param {Point} spot */
    const taken = (spot) => answers.some((n) => Math.abs(n.x - spot.x) < GRID.colGap / 2 && Math.abs(n.y - spot.y) < GRID.rowGap / 2);
    let k = 0;
    while (taken(answerSpot(first, k))) k += 1;
    return answerSpot(first, k);
  }

  /** @param {Tab} tab @param {Origin} origin */
  function plan(tab, origin) {
    if (tab.fold) return foldPlan(tab);
    const layout = gridLayout(tab.files, origin, roundsOf(tab));
    const first = layout.get(RUN_ANSWER);
    if (!tab.run || !first) return layout;
    tab.run.answers.forEach((_, k) => layout.set(answerId(k), answerSpot(first, k)));
    return layout;
  }

  /** @param {Tab} tab */
  function contentBounds(tab) {
    const widest = Math.max('structure'.length, ...tab.files.map((f) => fileLabel(f).length));
    // Once there is a run, an answer's label is its question and model, often wider than any file's.
    /** @param {string} id */
    const width = (id) => (tab.run && splitId(id)[0] === 'answer' ? nodeName(tab, id).length : widest) * charPx + NODE_PAD_PX;
    const b = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
    for (const [id, p] of tab.layout) {
      b.minX = Math.min(b.minX, p.x - width(id) / 2);
      b.maxX = Math.max(b.maxX, p.x + width(id) / 2);
      b.minY = Math.min(b.minY, p.y - BRIDGE_CLEARANCE);
      b.maxY = Math.max(b.maxY, p.y + iconHalf(splitId(id)[0]) + 2 * LABEL_LINE_PX);
    }
    for (const n of tab.nodes.values()) {
      const halfW = n.el.offsetWidth / 2;
      b.minX = Math.min(b.minX, n.x - halfW);
      b.maxX = Math.max(b.maxX, n.x + halfW);
      b.minY = Math.min(b.minY, n.y - iconHalf(n.kind));
      b.maxY = Math.max(b.maxY, n.y - iconHalf(n.kind) + n.el.offsetHeight);
    }
    return Number.isFinite(b.minX) ? b : { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  }

  function fit(animate = false) {
    const { view } = state.shown;
    const pw = panel.clientWidth;
    const ph = panel.clientHeight;
    if (!pw || !ph) return;
    const b = contentBounds(state.shown);
    const bw = Math.max(1, b.maxX - b.minX);
    const bh = Math.max(1, b.maxY - b.minY);
    const k = clamp(Math.min((pw - 2 * FIT_MARGIN) / bw, (ph - 2 * FIT_MARGIN) / bh), ZOOM.min, ZOOM.fitMax);
    view.k = k;
    view.x = (pw - bw * k) / 2 - b.minX * k;
    view.y = (ph - bh * k) / 2 - b.minY * k;
    if (animate) glide();
    applyView();
  }

  function fitShown(animate = false) {
    if (!state.shown.userMoved) fit(animate);
  }

  function refit() {
    state.shown.userMoved = false;
    fit(true);
  }

  // ---------- panel: nodes and edges ----------

  /** @param {NodeKind} kind @param {boolean} settled */
  function iconFor(kind, settled) {
    if (kind === 'doc') return ICONS.document;
    if (isRound(kind)) return ICONS.snippet;
    if (kind === 'answer') return `<span class="glyph" aria-hidden="true">${MARK}</span>`;
    return ICONS.tree(kind === 'parsed' && settled);
  }

  /** @param {Tab} tab @param {string} id @param {NodeKind} kind @param {LoadedFile | null} file */
  function nodeLabels(tab, id, kind, file) {
    if (kind === 'schema' || kind === 'answer') return [nodeName(tab, id)];
    const label = fileLabel(/** @type {LoadedFile} */ (file));
    return kind === 'doc' || kind === 'parsed' ? [label] : [label, roundName(kind)];
  }

  /** @param {Tab} tab @param {PanelNode} node @param {number} [sentences] the highlighted ones, for a document */
  function ariaName(tab, node, sentences = 0) {
    const stopped = node.el.classList.contains('failed') ? ', stopped' : '';
    if (node.kind === 'answer') {
      const { question, model } = answerAt(runTab(tab).run, node.id);
      return `answer to ${question} · ${model}${stopped}`;
    }
    const what = isRound(node.kind) ? `${roundName(node.kind)} relevance snippet` : ARIA_KIND[node.kind];
    if (!node.file) return `${nodeName(tab, node.id)}, ${what}${stopped}`;
    if (node.kind !== 'doc') return `${stripExt(node.file.name)}, ${what}${stopped}`;
    return `${node.file.name}, ${ARIA_KIND.doc}${sentences ? `, ${plural(sentences, 'highlighted sentence')}` : ''}`;
  }

  /** @param {Tab} tab @param {string} id @param {NodeKind} kind @param {Point} pos @param {{ working?: boolean, file?: LoadedFile | null, enter?: boolean }} [options] */
  function createNode(tab, id, kind, pos, { working = false, file = null, enter = true } = {}) {
    const classes = ['node', kind, working && 'working', enter && animates(tab) && 'enter'].filter(Boolean).join(' ');
    const el = h(
      'div',
      { class: classes, tabindex: '0', role: 'button', 'data-id': id, 'aria-pressed': kind === 'answer' ? 'false' : null },
      h('span', { class: 'icon', html: iconFor(kind, false) }),
      kind === 'answer' ? null : h('span', { class: 'spin', 'aria-hidden': 'true' }, MARK),
      nodeLabels(tab, id, kind, file).map((text, i) => h('span', { class: i ? 'label' : 'label name' }, text)),
      kind === 'doc' ? h('span', { class: 'pill', hidden: true }) : null,
    );
    const node = { id, kind, file, x: pos.x, y: pos.y, el };
    tab.nodes.set(id, node);
    el.setAttribute('aria-label', ariaName(tab, node));
    placeNode(node);
    tab.nodesEl.append(el);
    return node;
  }

  /** @param {RunTab} tab @param {string} id */
  function addNode(tab, id) {
    const [kind, d] = splitId(id);
    if (tab.fold && isPerDocument(kind)) {
      setCall(tab.fold, id, 'working');
      const fold = showFolded(tab, id);
      if (fold) return fold;
    }
    const node = createNode(tab, id, kind, /** @type {Point} */ (tab.layout.get(id)), { working: true, file: isPerDocument(kind) ? tab.run.docs[d] : null });
    if (tab.fold) placeInRows(tab, node);
    return node;
  }

  /** @param {Tab} tab @param {LoadedFile} file @param {Point} [pos] */
  function addDocNode(tab, file, pos) {
    /** @type {PanelNode | null} */
    let last = null;
    for (const n of tab.nodes.values()) if (n.kind === 'doc' && (!last || n.y > last.y)) last = n;
    // A node redrawn at its old position (a run starting) is not new, so it doesn't animate in.
    return createNode(tab, docId(file), 'doc', pos ?? (last ? { x: last.x, y: last.y + GRID.rowGap } : tidyOrigin(0)), { file, enter: !pos });
  }

  /** @param {Tab} tab @param {string} id */
  function removeNode(tab, id) {
    if (openIn(tab) === id) closeInspector();
    if (hoveredIn(tab) === id) state.hovered = null;
    if (tab.nodes.get(id) === gesture.node) endGesture();
    tab.nodes.get(id)?.el.remove();
    tab.nodes.delete(id);
    tab.placed.delete(id);
  }

  /** @param {number} t */
  const easeOut = (t) => 1 - (1 - t) ** 3;

  /** @param {Tab} tab */
  function clearPanel(tab) {
    cancelAnimationFrame(tab.arrangeFrame);
    if (openIn(tab)) closeInspector();
    if (tab === state.shown) endGesture();
    for (const layer of Object.values(tab.layers)) layer.replaceChildren();
    tab.edges.clear();
    tab.nodesEl.replaceChildren();
    tab.nodes.clear();
    tab.placed.clear();
    tab.drawing.classList.remove('is-lit');
    tab.trail = null;
    if (hoveredIn(tab)) state.hovered = null;
    showTrail(tab);
  }

  // A slide left going in a tab that is no longer shown jumps to where it was going.
  /** @param {Tab} tab */
  function finishSlide(tab) {
    if (!tab.arrangeFrame) return;
    cancelAnimationFrame(tab.arrangeFrame);
    tab.arrangeFrame = 0;
    for (const node of tab.nodes.values()) placeNode(node);
    redrawEdges(tab);
  }

  // Document nodes are created only here, or by a folded tab's rows: one per file of the tab, at its position in
  // `at` if it had one, else below the last document node. A run's grid is planned once, when it starts.
  /** @param {Tab} tab @param {{ at?: Map<string, Point>, animate?: boolean, replan?: boolean }} [options] */
  function syncDocNodes(tab, { at = new Map(), animate = false, replan = !tab.run } = {}) {
    const folded = Boolean(tab.fold);
    if (!tab.run) foldFiles(tab);
    if (tab.fold && !tab.run) refold(tab, { was: at, fit: false });
    if (!tab.fold) syncFileNodes(tab, at, folded ? 'tidy' : replan && 'own');
    if (tab !== state.shown) return;
    updateHint();
    fitShown(animate);
  }

  // A node per file: those of files gone, and of a fold, go. Planned again from where the nodes are, or, once a fold
  // gives way to its files, in a tidy column they slide to.
  /** @param {Tab} tab @param {Map<string, Point>} at @param {'own' | 'tidy' | false} replan */
  function syncFileNodes(tab, at, replan) {
    const keep = new Set(tab.files.map(docId));
    for (const n of [...tab.nodes.values()]) {
      if (n.kind !== 'doc' || (keep.has(n.id) && !n.fold)) continue;
      removeNode(tab, n.id);
      tab.layout.delete(n.id);
    }
    for (const f of tab.files) {
      if (tab.nodes.has(docId(f))) continue;
      const node = addDocNode(tab, f, at.get(docId(f)));
      tab.layout.set(node.id, { x: node.x, y: node.y });
    }
    if (replan === 'own') tab.layout = plan(tab, nodeOrigin(tab));
    if (replan !== 'tidy') return;
    tab.layout = plan(tab, tidyOrigin);
    slideToLayout(tab, () => {});
  }

  /** @param {PanelNode} node */
  function placeNode(node) {
    node.el.style.left = `${node.x}px`;
    node.el.style.top = `${node.y}px`;
  }

  /** @param {PanelNode} node */
  function drawnAt(node) {
    return { x: parseFloat(node.el.style.left), y: parseFloat(node.el.style.top) };
  }

  function arrange() {
    const tab = state.shown;
    tab.layout = plan(tab, tidyOrigin);
    tab.userMoved = false;
    tab.placed.clear();
    slideToLayout(tab);
  }

  // Each node but those the user placed slides from where it is drawn to its place in the tab's layout, then
  // `settled` runs. The model takes the final positions at once; only the drawing slides there.
  /** @param {Tab} tab @param {() => void} [settled] */
  function slideToLayout(tab, settled = () => fitShown(true)) {
    cancelAnimationFrame(tab.arrangeFrame);
    /** @type {{ node: PanelNode, from: Point }[]} */
    const moves = [];
    for (const node of tab.nodes.values()) {
      const to = tab.layout.get(node.id);
      if (!to || tab.placed.has(node.id)) continue;
      const from = drawnAt(node);
      Object.assign(node, to);
      if (from.x !== to.x || from.y !== to.y) moves.push({ node, from });
    }
    // Each frame draws again only the lines of nodes that move; the last draws every line, as a line hops the nodes
    // between its ends, and they may have moved.
    const moved = new Set(moves.map(({ node }) => node.id));
    const lines = [...tab.edges.values()].filter((e) => moved.has(e.from) || moved.has(e.to));
    const finish = () => {
      tab.arrangeFrame = 0;
      for (const { node } of moves) placeNode(node);
      redrawEdges(tab);
      positionInspector();
      settled();
    };
    if (!moves.length || !animates(tab)) return finish();
    const start = performance.now();
    const grabbed = new Set();
    /** @param {number} now */
    const step = (now) => {
      const t = Math.min(1, (now - start) / ARRANGE_MS);
      if (t === 1) return finish();
      const drawn = new Map();
      for (const { node, from } of moves) {
        if (gesture.moved && node === gesture.node) grabbed.add(node);
        if (grabbed.has(node)) continue;
        const at = { x: from.x + (node.x - from.x) * easeOut(t), y: from.y + (node.y - from.y) * easeOut(t) };
        drawn.set(node, at);
        node.el.style.left = `${at.x}px`;
        node.el.style.top = `${at.y}px`;
      }
      redrawEdges(tab, (n) => drawn.get(n) ?? n, lines);
      tab.arrangeFrame = requestAnimationFrame(step);
    };
    tab.arrangeFrame = requestAnimationFrame(step);
  }

  /** @param {Tab} tab @param {string} id */
  function markWorking(tab, id) {
    if (tab.fold) {
      setCall(tab.fold, id, 'working');
      if (showFolded(tab, id)) return;
    }
    drawnNode(tab, id).el.classList.add('working');
  }

  // A node is settled when its call finished, empty or not, or failed when the run stopped first.
  /** @param {Tab} tab @param {string} id */
  function settleNode(tab, id, { empty = false, failed = false } = {}) {
    if (tab.fold && settleFolded(tab, id, { empty, failed })) return;
    drawSettled(tab, drawnNode(tab, id), { empty, failed });
  }

  /** @param {Tab} tab @param {PanelNode} node @param {{ empty: boolean, failed: boolean }} how */
  function drawSettled(tab, node, { empty, failed }) {
    node.el.classList.remove('working');
    node.el.classList.toggle('void', empty);
    node.el.classList.toggle('failed', failed);
    node.el.querySelectorAll('.spin, .glyph').forEach((g) => {
      g.textContent = MARK;
    });
    if (node.kind === 'parsed' && !empty && !failed) /** @type {Element} */ (node.el.querySelector('.icon')).innerHTML = iconFor('parsed', true);
    if (!node.fold) node.el.setAttribute('aria-label', ariaName(tab, node));
    rerenderIfOpen(tab, node.id);
  }

  // Ids repeat across tabs; the popup and the hover are only ever on the shown tab's nodes.
  /** @param {Tab} tab */
  const openIn = (tab) => (tab === state.shown ? state.selected : null);
  /** @param {Tab} tab */
  const hoveredIn = (tab) => (tab === state.shown ? state.hovered : null);

  /** @param {Tab} tab @param {string} id */
  function rerenderIfOpen(tab, id) {
    if (openIn(tab) === id) renderInspector();
  }

  // A node has landed once its call finished without the run or ask stopping first.
  /** @param {Tab} tab @param {string} id */
  function landed(tab, id) {
    const classes = tab.nodes.get(id)?.el.classList;
    return Boolean(classes && !classes.contains('working') && !classes.contains('failed'));
  }

  /** @param {Tab} tab @param {PanelNode} a @param {PanelNode} b @param {(node: PanelNode) => Point} at */
  function blockers(tab, a, b, at) {
    const pa = at(a);
    const pb = at(b);
    const lo = Math.min(pa.x + iconHalf(a.kind), pb.x + iconHalf(b.kind));
    const hi = Math.max(pa.x - iconHalf(a.kind), pb.x - iconHalf(b.kind));
    let count = 0;
    for (const n of tab.nodes.values()) {
      if (n === a || n === b) continue;
      const p = at(n);
      if (p.x > lo && p.x < hi && Math.abs(p.y - (pa.y + pb.y) / 2) < iconHalf(n.kind) * 2) count += 1;
    }
    return count;
  }

  // One lane higher per blocker, so bridges from one node nest.
  /** @param {Tab} tab @param {PanelNode} a @param {PanelNode} b @param {(node: PanelNode) => Point} [at] */
  function edgePath(tab, a, b, at = (n) => n) {
    const pa = at(a);
    const pb = at(b);
    const sx = pa.x + iconHalf(a.kind) + EDGE_GAP;
    const tx = pb.x - iconHalf(b.kind) - EDGE_GAP;
    const hops = Math.abs(pa.y - pb.y) < ICON_HALF && tx > sx ? blockers(tab, a, b, at) : 0;
    if (!hops) {
      const c = Math.max(MIN_CURVE, Math.abs(tx - sx) / 2);
      return `M${sx} ${pa.y} C${sx + c} ${pa.y} ${tx - c} ${pb.y} ${tx} ${pb.y}`;
    }
    // A lane for each node the longest bridge hops: a document's dotted line to its parse, over each round and the structure.
    const lane = LANE_ROOM / (roundsOf(tab) + 1);
    const top = Math.min(pa.y, pb.y) - ICON_HALF - lane * hops;
    const ramp = Math.min(BRIDGE_RAMP / hops, (tx - sx) / 4);
    return `M${sx} ${pa.y} Q${sx + ramp} ${top} ${sx + 2 * ramp} ${top} H${tx - 2 * ramp} Q${tx - ramp} ${top} ${tx} ${pb.y}`;
  }

  /** @param {Tab} tab @param {EdgeKind} kind */
  function layerFor(tab, kind) {
    return kind === 'context' || kind === 'self' ? tab.layers.base : tab.layers.mid;
  }

  // A tab drawn while hidden would play every animation at once when shown.
  /** @param {Tab} tab */
  const animates = (tab) => tab === state.shown && !reduceMotion();

  // An element put back in the page plays its animation again while the class is on: a line moved to the
  // lit layer, or a drawing shown again.
  /** @param {Element} el */
  function endAnimation(el) {
    el.classList.remove('enter', 'draw', 'fade');
    el.removeAttribute('pathLength');
  }

  /** @param {Tab} tab @param {Element} el @param {'draw' | 'fade'} how */
  function animateEdge(tab, el, how) {
    if (!animates(tab)) return;
    endAnimation(el);
    if (how === 'draw') el.setAttribute('pathLength', '1');
    el.classList.add(how);
  }

  // Faint lines touch nearly every node; lighting them would light the panel.
  /** @param {Tab} tab @param {PanelEdge} edge */
  function lightsWith(tab, edge) {
    const id = hoveredIn(tab) ?? openIn(tab);
    return edge.kind !== 'context' && id !== null && (edge.from === id || edge.to === id);
  }

  /** @param {Tab} tab @param {string} from @param {string} to @param {EdgeKind} kind @param {'draw' | 'fade'} [how] */
  function addEdge(tab, from, to, kind, how) {
    // On a folded panel a line runs between the nodes that show its ends, once for all the lines a fold stands for.
    if (tab.fold) {
      const ends = shownEnds(tab, from, to);
      if (!ends) return null;
      [from, to] = ends;
    }
    const key = `${from}>${to}`;
    let edge = tab.edges.get(key);
    if (edge && tab.fold && EDGE_RANK[edge.kind] >= EDGE_RANK[kind]) return edge;
    if (!edge) {
      edge = { key, from, to, kind, el: document.createElementNS(SVG_NS, 'path') };
      edge.el.classList.add('edge');
      edge.el.dataset.key = key;
      tab.edges.set(key, edge);
    }
    edge.el.classList.remove(edge.kind);
    edge.kind = kind;
    edge.el.classList.add(kind);
    layerFor(tab, kind).append(edge.el);
    edge.el.setAttribute('d', edgePath(tab, drawnNode(tab, from), drawnNode(tab, to)));
    if (how) animateEdge(tab, edge.el, how);
    edge.el.classList.toggle('hl', lightsWith(tab, edge));
    return edge;
  }

  /** @param {Tab} tab @param {PanelEdge} edge */
  function dropEdge(tab, edge) {
    edge.el.remove();
    tab.edges.delete(edge.key);
  }

  /** @param {Tab} tab @param {(node: PanelNode) => Point} [at] @param {Iterable<PanelEdge>} [edges] */
  function redrawEdges(tab, at, edges = tab.edges.values()) {
    for (const e of edges) e.el.setAttribute('d', edgePath(tab, drawnNode(tab, e.from), drawnNode(tab, e.to), at));
  }

  function refreshHighlight() {
    const tab = state.shown;
    for (const e of tab.edges.values()) e.el.classList.toggle('hl', lightsWith(tab, e));
    for (const n of tab.nodes.values()) n.el.classList.toggle('selected', n.id === state.selected);
  }

  // Lights the trail of the tab's lit answer; an answer that cites nothing lights nothing.
  /** @param {RunTab} tab */
  function lightPath(tab) {
    const { run } = tab;
    const trail = run.lit ? trailOf(run, run.lit) : null;
    tab.trail = trail;
    tab.drawing.classList.toggle('is-lit', Boolean(trail));
    for (const [id, n] of tab.nodes) {
      n.el.classList.toggle('on-path', Boolean(trail?.nodes.has(id)));
      if (n.kind === 'answer') n.el.setAttribute('aria-pressed', String(id === run.lit));
    }
    run.docs.forEach((file, d) => {
      const node = tab.nodes.get(docId(file));
      if (node) countSentences(tab, node, trail?.quoted[d].sentences.length ?? 0);
    });
    for (const e of tab.edges.values()) {
      const on = Boolean(trail?.edges.has(e.key));
      const layer = on ? tab.layers.lit : layerFor(tab, e.kind);
      e.el.classList.toggle('lit', on);
      if (e.el.parentNode !== layer) layer.append(e.el);
    }
    showTrail(tab);
  }

  /** @param {Tab} tab @param {PanelNode} node @param {number} sentences */
  function countSentences(tab, node, sentences) {
    const pill = /** @type {HTMLElement} */ (node.el.querySelector('.pill'));
    pill.hidden = !sentences;
    pill.textContent = String(sentences);
    pill.title = plural(sentences, 'highlighted sentence');
    node.el.setAttribute('aria-label', ariaName(tab, node, sentences));
  }

  /** @type {Map<Element, { tab: Tab, draw: (trail: Trail | null) => Child }>} */
  const trailParts = new Map();

  // Fills `el` from the tab's trail now, and again in place as the trail changes, so that what is open, scrolled or
  // focused in it stays.
  /** @template {Element} T @param {Tab} tab @param {T} el @param {(trail: Trail | null) => Child} draw @returns {T} */
  function trailPart(tab, el, draw) {
    trailParts.set(el, { tab, draw });
    fill(el, draw(tab.trail));
    return el;
  }

  /** @param {Tab} tab */
  function showTrail(tab) {
    for (const [el, part] of trailParts) {
      if (!el.isConnected) trailParts.delete(el);
      else if (part.tab === tab) fill(el, part.draw(tab.trail));
    }
    if (tab === state.shown) markFiles();
  }

  // Only the lit answer draws faint lines from what it was shown; every answer keeps its cite lines.
  /** @param {RunTab} tab @param {string} id */
  function lightAnswer(tab, id) {
    refoldNow(tab);
    for (const e of [...tab.edges.values()]) {
      if (e.kind === 'context' && e.to !== id && splitId(e.to)[0] === 'answer') dropEdge(tab, e);
    }
    for (const from of shownInputs(tab, id)) {
      if (!tab.edges.has(`${from}>${id}`)) addEdge(tab, from, id, 'context');
    }
    tab.run.lit = id;
    lightPath(tab);
  }

  // A landing ask is the newest answer, so it is lit, unless another answer that has landed is open there.
  /** @param {RunTab} tab @param {string} id */
  function lightLanded(tab, id) {
    const open = openIn(tab);
    if (open && open !== id && splitId(open)[0] === 'answer' && landed(tab, open)) return;
    lightAnswer(tab, id);
  }

  // A run lights its own answer, or again the one lit before a replay.
  /** @param {RunTab} tab */
  const lightRun = (tab) => lightAnswer(tab, tab.run.lit ?? RUN_ANSWER);

  // ---------- panel: folding ----------
  // A panel of hundreds of rows is unreadable, so a tab of more than FOLD_FROM_DOCS documents folds those with nothing
  // relevant, as a diff folds unchanged lines. Only a snippet, or the answer's trail, gives a document its own row: a
  // run whose model parses every document stays folded.

  /** @param {NodeKind} kind @param {FoldRow} row */
  const foldId = (kind, row) => `${kind}-f${row.start}`;
  /** @param {FoldRow} row */
  const foldDocs = (row) => range(row.end - row.start + 1).map((k) => row.start + k);
  // A call is kept under its node's id, and reading a file under the document's own node.
  /** @param {Folding} fold @param {NodeKind} kind @param {number} d */
  const callId = (fold, kind, d) => (kind === 'doc' ? docId(fold.files[d]) : `${kind}-${d}`);
  /** @param {Tab} tab */
  const foldingOf = (tab) => /** @type {Folding} */ (tab.fold);

  /** @param {Tab} tab */
  function startFolding(tab) {
    if (tab.fold) cancelAnimationFrame(tab.fold.frame);
    const files = tab.run?.docs ?? tab.files;
    const n = files.length;
    if (n <= FOLD_FROM_DOCS) {
      tab.fold = null;
      return;
    }
    const kept = new Set(range(CONTEXT_DOCS).flatMap((k) => [k, n - 1 - k]));
    const index = new Map(files.map((f, d) => [docId(f), d]));
    const sets = { relevant: new Set(), kept, opened: new Set() };
    tab.fold = { files, index, calls: new Map(), settled: new Map(), ...sets, stopped: false, rows: [], closedRows: 0, foldOf: [], frame: 0 };
    foldRows(tab.fold);
  }

  // Before a run the files fold again as they change, and a fold the user opened stays open.
  /** @param {Tab} tab */
  function foldFiles(tab) {
    const was = tab.fold;
    if (was?.files === tab.files) return;
    startFolding(tab);
    if (!tab.fold || !was) return;
    const { opened, index } = tab.fold;
    for (const d of was.opened) {
      const now = index.get(docId(was.files[d]));
      if (now !== undefined) opened.add(now);
    }
    foldRows(tab.fold);
  }

  /** @param {Tab} tab @param {string} id */
  function docOf(tab, id) {
    const [kind, d] = splitId(id);
    if (kind === 'doc') return tab.fold?.index.get(id) ?? null;
    return isPerDocument(kind) && Number.isInteger(d) ? d : null;
  }

  // The fold a document's node is drawn on, if it is folded.
  /** @param {Tab} tab @param {string} id */
  function foldHolding(tab, id) {
    const d = docOf(tab, id);
    return (d === null ? null : tab.fold?.foldOf[d]) ?? null;
  }

  // The node that shows `id`: its own, or its fold's.
  /** @param {Tab} tab @param {string} id */
  function shownAs(tab, id) {
    const row = foldHolding(tab, id);
    return row ? foldId(splitId(id)[0], row) : id;
  }

  // A line's ends on a folded panel, or null when the nodes that show them are one, or not drawn.
  /** @param {Tab} tab @param {string} from @param {string} to */
  function shownEnds(tab, from, to) {
    const [a, b] = [shownAs(tab, from), shownAs(tab, to)];
    return a !== b && tab.nodes.has(a) && tab.nodes.has(b) ? [a, b] : null;
  }

  /** @param {Folding} fold */
  function foldRows(fold) {
    const n = fold.files.length;
    /** @param {number} d */
    const own = (d) => fold.relevant.has(d) || fold.kept.has(d);
    fold.rows = [];
    fold.foldOf = Array(n).fill(null);
    let opened = 0;
    for (let d = 0; d < n; ) {
      let end = d;
      while (!own(d) && end + 1 < n && !own(end + 1)) end += 1;
      if (own(d) || end - d + 1 < MIN_FOLD) fold.rows.push(...range(end - d + 1).map((k) => ({ doc: d + k })));
      else opened += addFold(fold, d, end);
      d = end + 1;
    }
    fold.closedRows = fold.rows.length - opened;
  }

  // A fold is open while any of its documents is, so it stays open as documents join it, and they open with it.
  // Returns how many rows it opened.
  /** @param {Folding} fold @param {number} start @param {number} end */
  function addFold(fold, start, end) {
    const docs = range(end - start + 1).map((k) => start + k);
    /** @type {FoldRow} */
    const row = { start, end, open: docs.some((d) => fold.opened.has(d)) };
    fold.rows.push(row);
    if (!row.open) {
      for (const d of docs) fold.foldOf[d] = row;
      return 0;
    }
    for (const d of docs) fold.opened.add(d);
    fold.rows.push(...docs.map((doc) => ({ doc })));
    return docs.length;
  }

  // The answers centre on the rows as they'd be with every fold closed, so that opening one doesn't move them.
  /** @param {Tab} tab */
  function foldPlan(tab) {
    const fold = foldingOf(tab);
    const rounds = roundsOf(tab);
    /** @param {NodeKind} kind */
    const x = (kind) => column(kind, rounds) * GRID.colGap;
    /** @type {PerDocumentKind[]} */
    const perDocument = [...roundKinds(rounds), 'parsed'];
    /** @type {Map<string, Point>} */
    const layout = new Map();
    fold.rows.forEach((row, i) => {
      const y = i * GRID.rowGap;
      if ('doc' in row) {
        layout.set(docId(fold.files[row.doc]), { x: 0, y });
        for (const kind of perDocument) layout.set(`${kind}-${row.doc}`, { x: x(kind), y });
        return;
      }
      layout.set(foldId('doc', row), { x: 0, y });
      if (!row.open) for (const kind of perDocument) layout.set(foldId(kind, row), { x: x(kind), y });
    });
    layout.set('schema', { x: x('schema'), y: 0 });
    const first = { x: x('answer'), y: ((fold.closedRows - 1) * GRID.rowGap) / 2 };
    range(tab.run?.answers.length ?? 1).forEach((k) => layout.set(answerId(k), answerSpot(first, k)));
    return layout;
  }

  // Top to bottom, as the page orders them for the keyboard. A column's node is drawn once a call there has started.
  /** @param {Tab} tab */
  function foldedNodes(tab) {
    const fold = foldingOf(tab);
    /** @type {PerDocumentKind[]} */
    const kinds = [...roundKinds(roundsOf(tab)), 'parsed'];
    /** @type {Map<string, { kind: NodeKind, doc: number, row?: undefined } | { kind: NodeKind, row: FoldRow }>} */
    const want = new Map();
    for (const row of fold.rows) {
      if ('doc' in row) {
        want.set(docId(fold.files[row.doc]), { kind: 'doc', doc: row.doc });
        for (const kind of kinds) if (fold.calls.has(`${kind}-${row.doc}`)) want.set(`${kind}-${row.doc}`, { kind, doc: row.doc });
        continue;
      }
      want.set(foldId('doc', row), { kind: 'doc', row });
      if (row.open) continue;
      for (const kind of kinds) if (foldDocs(row).some((d) => fold.calls.has(`${kind}-${d}`))) want.set(foldId(kind, row), { kind, row });
    }
    return want;
  }

  /** @param {Tab} tab @param {string} id @param {NodeKind} kind @param {FoldRow} row @param {Point} pos @param {boolean} enter */
  function createFoldNode(tab, id, kind, row, pos, enter) {
    const el = h(
      'div',
      { class: ['node', 'fold', kind, enter && animates(tab) && 'enter'].filter(Boolean).join(' '), tabindex: '0', role: 'button', 'data-id': id },
      h('span', { class: 'icon', html: kind === 'doc' ? ICONS.fold : iconFor(kind, false) }),
      h('span', { class: 'spin', 'aria-hidden': 'true' }, MARK),
      h('span', { class: 'label name' }),
      h('span', { class: 'label' }),
    );
    /** @type {PanelNode} */
    const node = { id, kind, file: null, x: pos.x, y: pos.y, el, fold: row };
    tab.nodes.set(id, node);
    placeNode(node);
    tab.nodesEl.append(el);
    return node;
  }

  // Whether a fold's documents are read: all back from round 1 with nothing, or the run stopped first, or not yet.
  /** @param {Tab} tab @param {FoldRow} row */
  function foldStatus(tab, row) {
    if (!tab.run) return '';
    const fold = foldingOf(tab);
    if (foldDocs(row).every((d) => fold.calls.get(`${roundKind(1)}-${d}`) === 'done')) return 'nothing relevant';
    return fold.stopped ? 'stopped' : 'reading…';
  }

  // Before a run a fold counts its files; in a run, its first node says whether they are read.
  /** @param {Tab} tab @param {PanelNode} node */
  function labelFold(tab, node) {
    const row = /** @type {FoldRow} */ (node.fold);
    const count = plural(row.end - row.start + 1, tab.run ? 'document' : 'more document');
    const status = foldStatus(tab, row);
    const [name, sub] = /** @type {NodeListOf<HTMLElement>} */ (node.el.querySelectorAll('.label'));
    name.textContent = count;
    sub.textContent = node.kind === 'doc' ? [status, row.open && 'fold'].filter(Boolean).join(' · ') : isRound(node.kind) ? roundName(node.kind) : '';
    sub.hidden = !sub.textContent;
    const column = isRound(node.kind) ? roundName(node.kind) : node.kind === 'parsed' ? 'parsed' : '';
    const stopped = node.el.classList.contains('failed') && status !== 'stopped' && 'stopped';
    node.el.setAttribute('aria-label', [count, column, row.open ? 'unfolded' : 'folded', status, stopped].filter(Boolean).join(', '));
    node.el.setAttribute('aria-expanded', String(row.open));
    node.el.title = `${row.open ? 'fold' : 'unfold'} ${count}`;
    node.el.classList.toggle('open', row.open);
  }

  // A fold's node is working while any of its calls is, stopped if any stopped, and empty only if each came back
  // empty. A node drawn as it stands already isn't drawn again, so that its open popup isn't built again.
  /** @param {Tab} tab @param {PanelNode} node @param {boolean} [fresh] */
  function paintNode(tab, node, fresh = false) {
    const fold = foldingOf(tab);
    const docs = node.fold ? foldDocs(node.fold) : [/** @type {number} */ (docOf(tab, node.id))];
    const calls = docs.map((d) => fold.calls.get(callId(fold, node.kind, d)));
    const classes = node.el.classList;
    if (calls.includes('working')) {
      classes.remove('void', 'failed');
      classes.add('working');
    } else if (calls.some(Boolean)) {
      const failed = calls.includes('failed');
      const back = (/** @type {number} */ d, /** @type {number} */ k) => calls[k] !== 'done' || isVoid(runTab(tab).run, `${node.kind}-${d}`);
      const empty = !failed && node.kind !== 'doc' && docs.every(back);
      const changed = fresh || classes.contains('working') || classes.contains('void') !== empty || classes.contains('failed') !== failed;
      if (changed) drawSettled(tab, node, { empty, failed });
    }
    if (!node.fold) return;
    labelFold(tab, node);
    const head = tab.nodes.get(foldId('doc', node.fold));
    if (head && head !== node) labelFold(tab, head);
  }

  /** @param {Folding} fold @param {string} id @param {CallState} call */
  function setCall(fold, id, call) {
    const was = fold.calls.get(id);
    fold.calls.set(id, call);
    const back = (/** @type {CallState | undefined} */ state) => state === 'done' || state === 'failed';
    if (back(was) === back(call)) return;
    const [kind] = splitId(id);
    fold.settled.set(kind, (fold.settled.get(kind) ?? 0) + (back(call) ? 1 : -1));
  }

  // Draws a call of a folded document on its fold; returns the fold's node, or null when the document has its own.
  /** @param {Tab} tab @param {string} id */
  function showFolded(tab, id) {
    const row = foldHolding(tab, id);
    if (!row) return null;
    const [kind] = splitId(id);
    const at = foldId(kind, row);
    let node = tab.nodes.get(at);
    if (!node) {
      node = createFoldNode(tab, at, kind, row, /** @type {Point} */ (tab.layout.get(at)), true);
      placeInRows(tab, node);
    }
    paintNode(tab, node);
    return node;
  }

  // A folded panel's page holds its nodes row by row, as the panel lays them out, so that the keyboard follows the
  // rows: a node drawn during a run goes after the last node before it in the layout.
  /** @param {Tab} tab @param {PanelNode} node */
  function placeInRows(tab, node) {
    /** @param {PanelNode} n */
    const spot = (n) => tab.layout.get(n.id) ?? n;
    const at = spot(node);
    /** @param {Point} a @param {Point} b */
    const precedes = (a, b) => a.y < b.y || (a.y === b.y && a.x < b.x);
    /** @type {PanelNode | null} */
    let before = null;
    for (const n of tab.nodes.values()) {
      if (n !== node && precedes(spot(n), at) && (!before || precedes(spot(before), spot(n)))) before = n;
    }
    if (before) before.el.after(node.el);
    else tab.nodesEl.prepend(node.el);
  }

  // Settles a call of a folded panel; true once that is drawn on its fold. A document with a snippet comes out of its
  // fold, and one kept in view goes into one once its first round is back with nothing, when the rows are next drawn.
  /** @param {Tab} tab @param {string} id @param {{ empty: boolean, failed: boolean }} how */
  function settleFolded(tab, id, { empty, failed }) {
    const fold = foldingOf(tab);
    const node = tab.nodes.get(id);
    if (node?.fold) {
      stopFold(tab, node, failed);
      return true;
    }
    const d = docOf(tab, id);
    if (d === null) return false;
    setCall(fold, id, failed ? 'failed' : 'done');
    const [kind] = splitId(id);
    const comesOut = isRound(kind) && !empty && !failed && !fold.relevant.has(d);
    const goesIn = kind === roundKind(1) && empty && !failed && fold.kept.has(d);
    if (comesOut) fold.relevant.add(d);
    if (comesOut || goesIn) {
      fold.kept.delete(d);
      refoldSoon(tab);
    }
    return Boolean(showFolded(tab, id));
  }

  // The documents on an answer's trail come out of their folds, so that the trail runs through their own rows.
  /** @param {Tab} tab @param {string} id */
  function unfoldTrail(tab, id) {
    const fold = foldingOf(tab);
    const trail = trailOf(runTab(tab).run, id);
    const docs = [...(trail?.nodes ?? [])].map((node) => docOf(tab, node)).filter((d) => d !== null && !fold.relevant.has(d));
    if (!docs.length) return;
    for (const d of docs) fold.relevant.add(/** @type {number} */ (d));
    refold(tab);
  }

  // The run stopped while a fold's calls were going: each of them stops too, or, reading a file, is done.
  /** @param {Tab} tab @param {PanelNode} node @param {boolean} failed */
  function stopFold(tab, node, failed) {
    const fold = foldingOf(tab);
    for (const d of foldDocs(/** @type {FoldRow} */ (node.fold))) {
      const id = callId(fold, node.kind, d);
      if (fold.calls.get(id) === 'working') setCall(fold, id, failed ? 'failed' : 'done');
    }
    paintNode(tab, node);
  }

  // A stopped run's folds say so, where their documents weren't all read.
  /** @param {Tab} tab */
  function foldsStopped(tab) {
    refoldNow(tab);
    foldingOf(tab).stopped = true;
    for (const node of tab.nodes.values()) if (node.fold) labelFold(tab, node);
  }

  // What a call was shown, as the panel draws it. On a folded panel, a call shown every document's output is drawn
  // shown each row's, and one shown the others' snippets, each relevant document's, which has a row of its own. The
  // documents of an opened fold draw no faint lines, in or out, so that opening hundreds doesn't draw a line from
  // each to each: their fold draws theirs while closed.
  /** @param {RunTab} tab @param {string} id */
  function shownInputs(tab, id) {
    const { run, fold } = tab;
    if (!fold) return inputsOf(run, id);
    const [kind, i] = splitId(id);
    /** @param {number} d */
    const opened = (d) => fold.opened.has(d) && !fold.relevant.has(d);
    if (isPerDocument(kind) && opened(i)) return [];
    const final = finalRound(run);
    /** @param {PerDocumentKind} step @param {(docs: number[]) => boolean} [keep] */
    const eachRow = (step, keep = () => true) =>
      fold.rows.flatMap((row) => {
        if ('doc' in row) return !opened(row.doc) && keep([row.doc]) ? [`${step}-${row.doc}`] : [];
        return !row.open && keep(foldDocs(row)) ? [foldId(step, row)] : [];
      });
    if (kind === 'parsed' || kind === 'schema') return eachRow(final);
    if (kind === 'answer') return [...eachRow(final), ...eachRow('parsed', (docs) => docs.some((d) => hasRecords(run.parsed[d])))];
    const previous = previousRound(kind);
    if (!previous) return [];
    const shown = [...fold.relevant].filter((j) => j !== i && summariesIn(run, previous)[j].summary);
    return shown.sort((a, b) => a - b).map((j) => `${previous}-${j}`);
  }

  // Every line a folded run draws: into each node, from what its call was shown, and from what each call back cites.
  /** @param {RunTab} tab */
  function foldedLines(tab) {
    const { run } = tab;
    const fold = foldingOf(tab);
    /** @type {Line[]} */
    const lines = [];
    for (const node of tab.nodes.values()) {
      if (isPerDocument(node.kind)) lines.push(...linesInto(tab, `${node.kind}-${node.fold ? node.fold.start : docOf(tab, node.id)}`));
    }
    if (tab.nodes.has('schema')) lines.push(...linesInto(tab, 'schema'));
    const lit = run.lit ?? RUN_ANSWER;
    if (tab.nodes.has(lit)) lines.push(...shownInputs(tab, lit).map((from) => /** @type {Line} */ ([from, lit, 'context'])));
    for (const [id, call] of fold.calls) if (call === 'done' && splitId(id)[0] !== 'doc') lines.push(...citeLines(tab, id));
    for (const node of tab.nodes.values()) if (node.kind === 'answer' && landed(tab, node.id)) lines.push(...citeLines(tab, node.id));
    return lines;
  }

  // A folded panel's lines once its rows change: those no longer wanted go and new ones are drawn, while the rest stay
  // as they are, drawn again only where their nodes slide.
  /** @param {Tab} tab */
  function redrawFoldedLines(tab) {
    /** @type {Map<string, Line>} */
    const wanted = new Map();
    for (const [from, to, kind] of tab.run ? foldedLines(runTab(tab)) : []) {
      const ends = shownEnds(tab, from, to);
      const key = ends && `${ends[0]}>${ends[1]}`;
      const had = key && wanted.get(key);
      if (ends && key && (!had || EDGE_RANK[had[2]] < EDGE_RANK[kind])) wanted.set(key, [ends[0], ends[1], kind]);
    }
    for (const e of [...tab.edges.values()]) if (wanted.get(e.key)?.[2] !== e.kind) dropEdge(tab, e);
    for (const [key, [from, to, kind]] of wanted) if (!tab.edges.has(key)) addEdge(tab, from, to, kind);
  }

  // Rows change as calls come back, often many to a frame: they are drawn again once a frame.
  /** @param {Tab} tab */
  function refoldSoon(tab) {
    const fold = foldingOf(tab);
    fold.frame ||= requestAnimationFrame(() => refold(tab));
  }

  // Rows waiting for their frame are drawn now, for what reads them.
  /** @param {Tab} tab */
  function refoldNow(tab) {
    if (tab.fold?.frame) refold(tab);
  }

  // A node not drawn now starts where it was drawn, or where `was` has it, and slides to its row. The focus on a node
  // that goes moves to the node that shows it now. With `fit`, the panel fits again unless the user moved it.
  /** @param {Tab} tab @param {{ fit?: boolean, was?: Map<string, Point> }} [options] */
  function refold(tab, { fit = true, was = new Map() } = {}) {
    const fold = foldingOf(tab);
    cancelAnimationFrame(fold.frame);
    fold.frame = 0;
    const drawn = new Map([...was, ...[...tab.nodes.values()].map((n) => /** @type {[string, Point]} */ ([n.id, drawnAt(n)]))]);
    const focused = focusedIn(tab);
    const folded = fold.foldOf;
    foldRows(fold);
    tab.layout = foldPlan(tab);
    fadeFaint(tab, fold.closedRows);
    const want = foldedNodes(tab);
    for (const n of [...tab.nodes.values()]) if ((n.kind === 'doc' || isPerDocument(n.kind)) && !want.has(n.id)) removeNode(tab, n.id);
    /** @type {PanelNode | null} */
    let before = null;
    for (const [id, w] of want) {
      const fresh = !tab.nodes.has(id);
      /** @type {PanelNode} */
      const node = tab.nodes.get(id) ?? drawFoldedNode(tab, id, w, drawn, folded, before);
      if (w.row) node.fold = w.row;
      paintNode(tab, node, fresh);
      before = node;
    }
    redrawFoldedLines(tab);
    if (tab.run?.lit && landed(tab, tab.run.lit)) lightPath(runTab(tab));
    if (focused && !tab.nodes.has(focused.id)) tab.nodes.get(shownAs(tab, focused.shows))?.el.focus({ preventScroll: true });
    slideToLayout(tab, () => {});
    if (fit && tab === state.shown) fitShown(true);
  }

  // Starts where what showed it was drawn: the fold it leaves, or the first row of a fold being made. Placed in the
  // page after the node `before` it, so that the keyboard reaches an opened fold's rows right after the fold.
  /**
   * @param {Tab} tab @param {string} id @param {{ kind: NodeKind, doc: number, row?: undefined } | { kind: NodeKind, row: FoldRow }} w
   * @param {Map<string, Point>} drawn @param {(FoldRow | null)[]} folded @param {PanelNode | null} before
   */
  function drawFoldedNode(tab, id, w, drawn, folded, before) {
    const fold = foldingOf(tab);
    const d = w.row ? w.row.start : w.doc;
    const at = drawn.get(folded[d] ? foldId(w.kind, folded[d]) : callId(fold, w.kind, d));
    const pos = at ?? /** @type {Point} */ (tab.layout.get(id));
    const node = w.row ? createFoldNode(tab, id, w.kind, w.row, pos, !at) : createNode(tab, id, w.kind, pos, { file: fold.files[w.doc], enter: !at });
    if (before) before.el.after(node.el);
    else tab.nodesEl.prepend(node.el);
    return node;
  }

  // The tab's node that has the focus, and the node of one document it shows: a fold shows its first.
  /** @param {Tab} tab */
  function focusedIn(tab) {
    const id = tab === state.shown && document.activeElement instanceof HTMLElement ? document.activeElement.dataset.id : undefined;
    const node = id === undefined ? undefined : tab.nodes.get(id);
    if (!node) return null;
    return { id: node.id, shows: node.fold ? callId(foldingOf(tab), node.kind, node.fold.start) : node.id };
  }

  // A fold opens, or folds again, in place. The view stays, but isn't marked as moved, so the panel fits again when it
  // next would; an opened fold low in the panel glides up, so that the rows it opened show.
  /** @param {Tab} tab @param {FoldRow} row */
  function toggleFold(tab, row) {
    const fold = foldingOf(tab);
    const opening = !row.open;
    for (const d of foldDocs(row)) {
      if (opening) fold.opened.add(d);
      else fold.opened.delete(d);
    }
    refold(tab, { fit: false });
    const head = tab.nodes.get(foldId('doc', row));
    if (!head) return;
    head.el.focus({ preventScroll: true });
    const { view } = tab;
    const top = panel.clientHeight * UNFOLD_TOP;
    if (!opening || view.y + head.y * view.k <= top) return;
    view.y = top - head.y * view.k;
    glide();
    applyView();
  }

  // ---------- inspector ----------

  /** @param {string[]} ids */
  function refs(ids) {
    if (!ids.length) return [h('span', { class: 'k' }, 'none')];
    return ids.map((id, i) => [
      i ? h('span', { class: 'k' }, ' · ') : null,
      state.shown.nodes.has(id)
        ? h('button', { class: 'ref', type: 'button', onclick: () => jumpTo(id) }, nodeName(state.shown, id))
        : h('span', { class: 'k' }, nodeName(state.shown, id)),
    ]);
  }

  /** @param {string} key @param {...Child} value */
  function row(key, ...value) {
    return h('div', { class: 'row' }, h('span', { class: 'k' }, key), ' ', value);
  }

  /** @param {string} text */
  function textBlock(text, empty = 'nothing relevant') {
    return text ? h('p', { class: 'insp-text' }, text) : h('p', { class: 'insp-text k' }, empty);
  }

  // A block per line of the JSON, its newline kept so it copies as is, wrapping under its own indent.
  /** @param {unknown} value */
  function jsonBlock(value) {
    const lines = JSON.stringify(value, null, JSON_INDENT).split(/(?<=\n)/);
    /** @param {string} line */
    const hanging = (line) => h('span', { style: `--indent: ${line.length - line.trimStart().length}ch` }, line);
    return [
      lines.length > PREVIEW_LINES ? h('div', { class: 'k' }, `first ${fmtInt(PREVIEW_LINES)} lines shown`) : null,
      h('pre', { class: 'code wrap json' }, lines.slice(0, PREVIEW_LINES).map(hanging)),
    ];
  }

  /** @param {Run} run @param {string} id */
  function callMeta(run, id) {
    const call = run.calls?.[id];
    return call ? h('div', { class: 'k' }, `${fmtInt(call.tokens)} tokens · ${fmtUsd(call.usd)}`) : null;
  }

  /** @param {LoadedFile} file @param {TextRange} shown */
  function fileMeta(file, shown) {
    const meta = [fmtBytes(file.size), file.sample ? 'sample memo' : 'your file'];
    if (shown.start) meta.push(`characters ${fmtInt(shown.start + 1)}–${fmtInt(shown.end)} shown`);
    else if (shown.end < file.text.length) meta.push(`first ${fmtInt(PREVIEW_CHARS)} characters shown`);
    return meta.join(' · ');
  }

  // A file's first PREVIEW_CHARS, or as many around a sentence past them, so that every sentence counted can be shown.
  /** @param {LoadedFile} file @param {TextRange | null} at */
  function shownPart(file, at) {
    const start = at && at.end > PREVIEW_CHARS ? Math.max(0, Math.min(at.start - PREVIEW_CHARS / 2, file.text.length - PREVIEW_CHARS)) : 0;
    return { start, end: Math.min(file.text.length, start + PREVIEW_CHARS) };
  }

  // The shown part of `text`, each span in `spans` marked, titled by `title`, and `at` flashed.
  /** @param {string} text @param {TextRange} shown @param {Span[]} spans @param {(span: Span) => string} title @param {TextRange | null} at */
  function markedText(text, shown, spans, title, at) {
    const cuts = cutsOf([shown, ...spans, ...(at ? [at] : [])]).filter((cut) => cut >= shown.start && cut <= shown.end);
    return cuts.slice(1).map((end, i) => {
      const span = spans.find((r) => r.start <= cuts[i] && r.end >= end);
      const flash = at && at.start <= cuts[i] && at.end >= end;
      if (!span && !flash) return text.slice(cuts[i], end);
      const classes = [span && span.by.length > 1 && 'multi', flash && 'flash'].filter(Boolean).join(' ');
      return h(span ? 'mark' : 'span', { class: classes || null, title: span ? title(span) : null }, text.slice(cuts[i], end));
    });
  }

  /** @param {Tab} tab @param {LoadedFile} file @param {TextRange | null} [at] a sentence to show */
  function markedFile(tab, file, at = null) {
    const shown = shownPart(file, at);
    /** @param {Trail | null} trail */
    const quotedOf = (trail) => trail?.quoted[tab.files.indexOf(file)];
    /** @param {Trail | null} trail */
    const line = (trail) => {
      const sentences = quotedOf(trail)?.sentences.length;
      const used = sentences ? `${plural(sentences, 'sentence')} used` : trail?.nodes.has(docId(file)) && 'used';
      return [fileMeta(file, shown), used && [' · ', h('span', { class: 'used' }, used)]];
    };
    const meta = trailPart(tab, h('div', { class: 'k' }), line);
    if (!file.text) return [meta, h('p', { class: 'k' }, 'This file has no readable text.')];
    // Spans come only with a trail.
    /** @param {Trail | null} trail */
    const text = (trail) => markedText(file.text, shown, quotedOf(trail)?.spans ?? [], (span) => quotedBy(tab, /** @type {Trail} */ (trail), span), at);
    return [meta, trailPart(tab, h('pre', { class: 'code wrap' }), text)];
  }

  /** @param {string} summary @param {Child} body @param {string} [className] */
  function fold(summary, body, className = 'fold') {
    return h('details', { class: className }, h('summary', {}, summary), h('div', { class: 'fold-body' }, body));
  }

  /** @param {Run} run @param {string} id */
  function provenanceFold(run, id) {
    const provenance = cites(run, id);
    return fold(`provenance ${provenance.length}`, refs(provenance), 'fold provenance');
  }

  /** @param {LoadedFile} file @param {TextRange} at @param {string} className @param {Child} [text] */
  function sentenceButton(file, at, className, text = null) {
    return h('button', { class: `ref ${className}`, type: 'button', title: 'show in the document', onclick: () => jumpTo(docId(file), at) }, text);
  }

  // A summary's quotes, marked while it is on the lit answer's trail; each one r3con found opens its document there.
  /** @param {RunTab} tab @param {RoundKind} kind @param {number} d */
  function quotesFold(tab, kind, d) {
    const { quotes } = summariesIn(tab.run, kind)[d];
    if (!quotes.length) return null;
    /** @param {RunQuote} quote */
    const item = (quote) => {
      if (!located(quote)) return h('p', {}, quote.text);
      return trailPart(tab, sentenceButton(tab.run.docs[d], quote, 'quote'), (trail) => (trail?.quoting.has(quote.summary) ? h('mark', {}, quote.text) : quote.text));
    };
    return fold(`quotes ${quotes.length}`, h('div', { class: 'quotes' }, quotes.map(item)));
  }

  /** @param {Tab} tab @param {string} id */
  function trailLine(tab, id) {
    return trailPart(tab, h('div', { class: 'trail-line' }), (trail) => {
      if (!trail) return null;
      if (!trail.nodes.has(id)) return h('span', { class: 'k' }, "not on the answer's trail");
      const citer = trail.citedBy.get(id);
      return h('span', { class: 'used' }, citer ? `on the answer's trail · cited by ${citerName(tab, citer)}` : "on the answer's trail");
    });
  }

  /** @param {LoadedFile} file @param {string} label */
  function openFileButton(file, label) {
    return h('button', { class: 'btn', type: 'button', onclick: () => openFile(file) }, label);
  }

  /** @param {LoadedFile} file @param {TextRange | null} at */
  function docBody(file, at) {
    const { run, nodes } = state.shown;
    const index = run ? run.docs.indexOf(file) : -1;
    const snippet = `${roundKind(1)}-${index}`;
    return [
      markedFile(state.shown, file, at),
      index >= 0 && nodes.has(snippet) ? row('relevance snippet', refs([snippet])) : null,
      openFileButton(file, 'open in files ›'),
    ];
  }

  // Round 1 reads each document alone, so only a later round's snippet has provenance.
  /** @param {RunTab} tab @param {RoundKind} kind @param {number} d */
  function snippetBody(tab, kind, d) {
    const { run } = tab;
    const id = `${kind}-${d}`;
    return [
      textBlock(summariesIn(run, kind)[d].summary),
      trailLine(tab, id),
      row('read', refs([docId(run.docs[d])])),
      quotesFold(tab, kind, d),
      roundOf(kind) > 1 ? provenanceFold(run, id) : null,
      callMeta(run, id),
      openFileButton(run.docs[d], 'open document ›'),
    ];
  }

  /** @param {string} id @param {TextRange | null} at */
  function inspectorBody(id, at) {
    const [kind, d] = splitId(id);
    const node = drawnNode(state.shown, id);
    if (kind === 'doc') return docBody(/** @type {LoadedFile} */ (node.file), at);
    const tab = runTab(state.shown);
    const { run } = tab;
    if (node.el.classList.contains('working')) return [h('p', { class: 'k' }, 'working…')];
    if (node.el.classList.contains('failed')) return [h('p', { class: 'k' }, kind === 'answer' && d > 0 ? 'the ask stopped before it answered' : 'the run stopped before this finished')];
    if (isRound(kind)) return snippetBody(tab, kind, d);
    if (kind === 'schema') {
      const schema = /** @type {Schema} */ (run.schema);
      return [
        schema.thought ? row('reasoning', schema.thought) : null,
        h('pre', { class: 'code' }, schema.code),
        callMeta(run, 'schema'),
      ];
    }
    if (kind === 'parsed') {
      const filled = Object.entries(run.parsed[d].parse).filter(([, records]) => records.length > 0);
      return [
        filled.length ? jsonBlock(Object.fromEntries(filled)) : textBlock('', 'nothing parsed'),
        row('read', refs([docId(run.docs[d])])),
        provenanceFold(run, id),
        row('used', refs(['schema'])),
        callMeta(run, id),
        openFileButton(run.docs[d], 'open document ›'),
      ];
    }
    const answer = answerOf(run, id);
    const trail = answerTrail(tab, id);
    const quoted = trail && sentencesFrom(trail);
    return [
      row('question', answer.question),
      answer.simulated ? h('div', { class: 'flag' }, SIMULATED_LABEL) : null,
      h('p', { class: 'insp-text strong' }, answer.text),
      provenanceFold(run, id),
      quoted && h('div', { class: 'used' }, quoted),
      h('div', { class: 'k' }, answer.summary),
    ];
  }

  // Scrolls the text `el` is in, which the popup shrinks to fit, to put `el` in its middle; scrollIntoView would also
  // scroll the panel, which only the view moves.
  /** @param {HTMLElement} el */
  function scrollToMiddle(el) {
    const box = /** @type {HTMLElement} */ (el.parentElement);
    const r = el.getBoundingClientRect();
    const top = box.scrollTop + r.top - box.getBoundingClientRect().top - (box.clientHeight - r.height) / 2;
    box.scrollTo({ top, behavior: reduceMotion() ? 'auto' : 'smooth' });
  }

  /** @param {TextRange | null} [at] a sentence to show, if the popup is a document's */
  function renderInspector(at = null) {
    const id = state.selected;
    if (!id || !state.shown.nodes.has(id)) return;
    const [kind] = splitId(id);
    inspector.setAttribute('aria-label', nodeName(state.shown, id));
    inspector.classList.toggle('wide', kind === 'schema' || kind === 'parsed');
    fill(
      inspector,
      h(
        'div',
        { class: 'insp-head' },
        h('div', { class: 'insp-title' }, h('span', { class: kind === 'answer' ? 'g accent' : 'g' }, '● '), nodeName(state.shown, id)),
        h('button', { class: 'btn', type: 'button', id: 'inspector-close', onclick: () => closeInspector(true) }, 'esc'),
      ),
      inspectorBody(id, at),
    );
    positionInspector();
    /** @type {HTMLElement | null} */
    const flashed = inspector.querySelector('.flash');
    if (flashed) scrollToMiddle(flashed);
  }

  // Selecting an answer that has landed also lights it, once its popup is built, which the light leaves alone.
  /** @param {string} id @param {{ focus?: boolean, at?: TextRange | null }} [options] */
  function openInspector(id, { focus = false, at = null } = {}) {
    const fold = state.shown.nodes.get(id)?.fold;
    if (fold) return toggleFold(state.shown, fold);
    state.selected = id;
    inspector.hidden = false;
    inspector.scrollTop = 0;
    renderInspector(at);
    if (splitId(id)[0] === 'answer' && landed(state.shown, id)) lightAnswer(runTab(state.shown), id);
    refreshHighlight();
    if (focus) inspector.focus({ preventScroll: true });
  }

  function closeInspector(restoreFocus = false) {
    if (inspector.hidden) return;
    const had = inspector.contains(document.activeElement);
    const node = state.shown.nodes.get(/** @type {string} */ (state.selected));
    inspector.hidden = true;
    state.selected = null;
    refreshHighlight();
    if (restoreFocus && had && node) node.el.focus({ preventScroll: true });
  }

  function positionInspector() {
    const { view } = state.shown;
    if (inspector.hidden) return;
    if (state.inspectorDroppedAt) return placeInspector(state.inspectorDroppedAt);
    const node = state.shown.nodes.get(/** @type {string} */ (state.selected));
    if (!node) return;
    const w = inspector.offsetWidth;
    const halfW = (node.el.offsetWidth / 2) * view.k;
    const cx = view.x + node.x * view.k;
    const cy = view.y + node.y * view.k;
    let left = cx + halfW + POPOVER_GAP;
    if (left + w > panel.clientWidth - POPOVER_MARGIN) left = cx - halfW - POPOVER_GAP - w;
    placeInspector({ left, top: cy - iconHalf(node.kind) * view.k - 6 });
  }

  /** @param {Offset} offset */
  function placeInspector({ left, top }) {
    const maxLeft = panel.clientWidth - inspector.offsetWidth - POPOVER_MARGIN;
    const maxTop = panel.clientHeight - inspector.offsetHeight - POPOVER_MARGIN;
    inspector.style.left = `${clamp(left, POPOVER_MARGIN, Math.max(POPOVER_MARGIN, maxLeft))}px`;
    inspector.style.top = `${clamp(top, POPOVER_MARGIN, Math.max(POPOVER_MARGIN, maxTop))}px`;
  }

  /** @param {string} id @param {TextRange | null} [at] */
  function jumpTo(id, at = null) {
    const { view } = state.shown;
    refoldNow(state.shown);
    // A node in a fold is shown by opening the fold first.
    const fold = state.shown.nodes.has(id) ? null : state.shown.nodes.get(shownAs(state.shown, id))?.fold;
    if (fold) toggleFold(state.shown, fold);
    const node = state.shown.nodes.get(id);
    if (!node) return;
    const pw = panel.clientWidth;
    const ph = panel.clientHeight;
    const sx = view.x + node.x * view.k;
    const sy = view.y + node.y * view.k;
    if (sx < JUMP_MARGIN || sx > pw - JUMP_MARGIN || sy < JUMP_MARGIN || sy > ph - JUMP_MARGIN) {
      view.x = pw / 2 - node.x * view.k;
      view.y = ph / 2 - node.y * view.k;
      state.shown.userMoved = true;
      glide();
      applyView();
    }
    openInspector(id, { at });
    node.el.focus({ preventScroll: true });
  }

  // ---------- chat ----------

  /** @param {string} kind @param {Child} text @param {LineOptions} [options] */
  function line(kind, text, { node = null, glyph = null, dim = false } = {}) {
    const g = glyph ?? { sub: '⎿', user: '›' }[kind] ?? '●';
    const classes = `line ${kind}${node ? ' link' : ''}${dim ? ' dim' : ''}`;
    return h(
      node ? 'button' : 'div',
      { class: classes, type: node ? 'button' : null, 'data-node': node },
      h('span', { class: 'g', 'aria-hidden': 'true' }, g),
      h('span', { class: 't' }, text),
    );
  }

  // A chat goes to its newest line in the next frame it can be seen in, but no further than puts an answer landed
  // since at its top, so that an answer taller than the chat keeps its start in view.
  /** @param {Tab} tab */
  function scrollChat(tab) {
    if (tab.chatToEnd) return;
    tab.chatToEnd = true;
    requestAnimationFrame(() => toNewestLine(tab));
  }

  // How far `chat` scrolls to put `el` where its first line sits, under its padding.
  /** @param {HTMLElement} chat @param {HTMLElement} el */
  const scrollTopFor = (chat, el) =>
    chat.scrollTop + el.getBoundingClientRect().top - chat.getBoundingClientRect().top - parseFloat(getComputedStyle(chat).paddingTop);

  /** @param {Tab} tab */
  function toNewestLine(tab) {
    const { chat, chatAnswer } = tab;
    if (!tab.chatToEnd || !chat.checkVisibility()) return;
    chat.scrollTop = chatAnswer ? Math.min(chat.scrollHeight, scrollTopFor(chat, chatAnswer)) : chat.scrollHeight;
    tab.chatToEnd = false;
    tab.chatAnswer = null;
  }

  // In the next frame, as the status line and the composer can still change the pane's height.
  /** @param {Tab} tab */
  function revealChat(tab) {
    if (tab.chatToEnd) requestAnimationFrame(() => toNewestLine(tab));
  }

  /** @param {Tab} tab @param {string} text */
  function tellChat(tab, text) {
    tab.chat.append(line('main', text, { dim: true }));
    scrollChat(tab);
  }

  // Only the shown tab's chat is in view, so its links are to the shown tab's nodes.
  /** @param {MouseEvent} e */
  function onChatClick(e) {
    /** @type {HTMLElement | null} */
    const link = targetOf(e).closest('button[data-node], button[data-offer]');
    if (!link) return;
    const { node, offer } = link.dataset;
    if (node) jumpTo(node);
    if (offer) prepareAsk(RECORDED_ASKS[Number(offer)]);
  }

  let nextLog = 0;

  /** @param {Tab} tab @param {HTMLElement} body @param {string} [first] */
  function narrator(tab, body, first = '') {
    const id = `log-${nextLog++}`;
    const lines = h('div', { class: 'log-lines', id, hidden: true, 'aria-live': 'off' });
    const latest = h('span', { class: 't', id: `${id}-latest` }, first);
    const asked = /** @type {Element} */ (body.parentElement?.querySelector('.line.user .t'));
    asked.id ||= `${id}-asked`;
    const toggleLines = () => {
      lines.hidden = !lines.hidden;
      toggle.ariaExpanded = String(!lines.hidden);
    };
    const labels = `${asked.id} ${latest.id}`;
    const toggle = h('button', { class: 'btn log-toggle', type: 'button', 'aria-expanded': 'false', 'aria-controls': id, 'aria-labelledby': labels, html: ICONS.chevron, onclick: toggleLines });
    const mark = h('span', { class: 'log-spin', 'aria-hidden': 'true' }, MARK);
    body.append(h('div', { class: 'log' }, h('div', { class: 'line log-head' }, toggle, mark, latest), lines));
    /** @type {Map<string, HTMLElement>} */
    const named = new Map();
    /** @param {HTMLElement} parent @param {string} kind @param {Child} text @param {LineOptions} [opts] */
    const add = (parent, kind, text, opts) => {
      const el = line(kind, text, opts);
      parent.append(el);
      if (!parent.hidden) scrollChat(tab);
      return el;
    };
    /** @param {string} kind @param {string} text @param {LineOptions} [opts] */
    const log = (kind, text, opts) => {
      if (kind === 'main') latest.textContent = text;
      return add(lines, kind, text, opts);
    };
    return {
      log,
      // Logs a line once under `key`; logging it again rewrites it in place.
      /** @param {string} key @param {string} kind @param {string} text @param {LineOptions} [opts] */
      logAs(key, kind, text, opts) {
        const said = named.get(key);
        if (said) /** @type {Element} */ (said.querySelector('.t')).textContent = text;
        else named.set(key, log(kind, text, opts));
      },
      /** @param {string} kind @param {Child} text @param {LineOptions} [opts] */
      say: (kind, text, opts) => add(body, kind, text, opts),
      // The r3con stage being told, READING while a replay reads its documents, or null between them.
      /** @type {string | null} */
      stage: null,
      /** @param {string} text @param {'ok' | 'err' | 'flag'} tone */
      finish(text, tone) {
        latest.textContent = text;
        mark.className = tone;
        mark.textContent = '●';
      },
    };
  }

  /** @param {Tab} tab @param {string} question */
  function addTurn(tab, question) {
    const body = h('div', { class: 'turn-body' });
    tab.chat.append(h('div', { class: 'turn' }, line('user', question), body));
    scrollChat(tab);
    return body;
  }

  // The status is a live region, so rewriting it unchanged, as on each keystroke, would talk over the user.
  /** @param {...(string | Node)} content */
  function setStatus(...content) {
    const text = content.map((part) => (typeof part === 'string' ? part : part.textContent)).join('');
    if ($('run-status').textContent !== text) $('run-status').replaceChildren(...content);
  }

  /** @param {string} text */
  const errorLine = (text) => [h('span', { class: 'err' }, '● '), text];
  /** @param {string} text */
  const errorStatus = (text) => setStatus(...errorLine(text));
  /** @param {Narrator} narr @param {string} model @param {boolean} simulated @param {Totals | null | undefined} totals */
  function sayReady(narr, model, simulated, totals) {
    // The cost alone, or the tokens where it is unknown, keeps the line short.
    const spend = spendOf(simulated, totals).slice(-1);
    narr.finish(['ready', model, ...spend].join(' · '), simulated ? 'flag' : 'ok');
  }
  /** @param {Narrator} narr @param {Ending} ending */
  function sayEnding(narr, { outcome, error }) {
    narr.finish(outcome, 'err');
    if (error) narr.say('main', h('span', { class: 'err' }, error));
  }
  /** @param {Run} run */
  const doneLine = (run) => [h('span', { class: run.simulated ? 'flag' : 'ok' }, '● '), `done · ${runSummary(run)}`];

  // A tab keeps its run's status, which the status line shows while the tab is shown; a stop's line stays
  // until what it stopped has ended. A tab without a run says what its next question would run.
  /** @param {Tab} tab */
  function showStatus(tab) {
    if (tab !== state.shown) return;
    if (!tab.run) return readyStatus();
    setStatus(...(stopping(tab) ? [STOPPING] : tab.status));
  }

  /** @param {Tab} tab @param {...(string | Node)} content */
  function tabStatus(tab, ...content) {
    tab.status = content;
    showStatus(tab);
  }

  /** @param {string} verb @param {string | null} stage */
  function inStage(verb, stage) {
    if (!stage) return verb;
    return stage === READING ? `${verb} while reading` : `${verb} in ${stage}`;
  }

  // A job the page asked to stop ends as stopped; any other failure gives its error.
  /** @param {boolean} stopped @param {{ stage: string | null, error: string }} event @returns {Ending} */
  const endingOf = (stopped, { stage, error }) => ({ outcome: inStage(stopped ? 'stopped' : 'failed', stage), error: stopped ? '' : error });

  function readyStatus() {
    if (state.shown.run) return;
    if (connectionProblem) return errorStatus(connectionProblem);
    if (!state.shown.files.length) return setStatus(ADD_FILES);
    setStatus(`${fileCount()} ready · ${mode.nextRun(questionEl.value.trim())}`);
  }

  // A question can ask over the shown tab's run once its answer has landed and the run has ended.
  const canAsk = () => landed(state.shown, RUN_ANSWER) && !mode.runGoing(state.shown);
  // Enter and the run button ask once there is a run to ask over, or one going; before any run, or after a
  // failed one, they run. The composer's labels follow this, so they say what a press does.
  const primaryAction = () => (canAsk() || mode.runGoing(state.shown) ? 'ask' : 'run');

  // Hiding or disabling the focused button would drop the focus to the page.
  /** @param {HTMLButtonElement} button @param {boolean} shown */
  function offer(button, shown, enabled = true) {
    if (!(shown && enabled) && document.activeElement === button) questionEl.focus();
    button.hidden = !shown;
    button.disabled = !enabled;
  }

  function updateComposer() {
    const asks = primaryAction() === 'ask';
    updateTabs();
    $('run-btn').textContent = asks ? 'ask ⏎' : 'run ⏎';
    // Stop stays enabled while the shown tab has something going that it hasn't asked to stop yet.
    offer($('stop-btn', HTMLButtonElement), tabBusy(state.shown), toStop(state.shown).length > 0 || !stopping(state.shown));
    offer($('new-run-btn', HTMLButtonElement), asks);
    $('replay-btn').hidden = live || !state.shown.run;
  }

  function updateHint() {
    const { run, files, removed, reading } = state.shown;
    const listed = removed?.files ?? files;
    const count = fileCount(listed);
    trash.hidden = Boolean(run) || !listed.length;
    // Not while a batch is read: its landing would end undo, and what remove all took couldn't be put back.
    const offered = !trash.hidden && !reading;
    $('panel-hint').hidden = Boolean(run);
    $('panel-hint-text').textContent = listed.length ? `${count}${removed ? ' removed' : ''}${offered ? ' · ' : ''}` : ADD_FILES;
    for (const button of removeAllButtons) {
      button.hidden = !offered;
      button.textContent = removed ? UNDO : REMOVE_ALL;
      button.ariaLabel = button.title = removed ? undoName(count) : REMOVE_ALL_NAME;
    }
  }

  // `action` is ask, over the run on the panel, or run, a new one over the files.
  /** @param {'ask' | 'run'} action */
  function submitQuestion(action) {
    const question = questionEl.value.trim();
    if (!question) {
      questionEl.focus();
      return;
    }
    const asks = action === 'ask';
    // An ask needs only the run's representation; a run needs files.
    const refused = asks || state.shown.files.length ? refusal(question, asks) : null;
    if (refused) {
      setStatus(refused);
      if (refused === TYPE_A_MODEL) modelOther.focus();
      return;
    }
    if (asks) {
      mode.ask(runTab(state.shown), question);
    } else if (!state.shown.files.length) {
      addTurn(state.shown, question).append(line('main', 'Add files in the files tab first, then ask again.', { dim: true }));
    } else {
      startRun(question);
    }
    questionEl.value = '';
    updateComposer();
  }

  // Why the question can't be sent yet, if it can't; it then stays in the box.
  /** @param {string} question @param {boolean} asks */
  function refusal(question, asks) {
    if (asks && mode.runGoing(state.shown)) return ASK_WHEN_DONE;
    if (!pickedModel() && mode.needsModel(question, asks)) return TYPE_A_MODEL;
    return asks ? null : mode.refuseFiles(state.shown.files);
  }

  // ---------- replay ----------

  /** @param {Tab} tab */
  function cancelTimers(tab) {
    tab.timers.forEach(clearTimeout);
    tab.timers = [];
  }

  /**
   * @param {(t: number, fn: () => void) => void} at
   * @param {number} t0
   * @param {number[]} order
   * @param {number} n
   * @param {(i: number) => void} appear
   * @param {(doc: number, rank: number) => void} settle
   */
  function stage(at, t0, order, n, appear, settle, pace = PACE.model) {
    const spread = TIMING.maxStageSpread / Math.max(1, n - 1);
    const appearGap = Math.min(pace.appearStagger, spread);
    const settleGap = Math.min(pace.settleStagger, spread);
    for (let i = 0; i < n; i++) at(t0 + i * appearGap, () => appear(i));
    let last = t0;
    order.forEach((doc, rank) => {
      last = Math.max(last + TIMING.minSettleGap, t0 + pace.settleFirst + rank * settleGap, t0 + doc * appearGap + pace.minWork);
      at(last, () => settle(doc, rank));
    });
    return last;
  }

  /** @param {Run} run @param {string} id */
  function inputsOf(run, id) {
    const [kind, i] = splitId(id);
    /** @param {string} step @param {(j: number) => unknown} [keep] */
    const every = (step, keep = () => true) => range(run.docs.length).filter(keep).map((j) => `${step}-${j}`);
    const final = finalRound(run);
    if (kind === 'parsed' || kind === 'schema') return every(final);
    if (kind === 'answer') return [...every(final), ...every('parsed', (j) => hasRecords(run.parsed[j]))];
    const previous = previousRound(kind);
    return previous ? every(previous, (j) => j !== i && summariesIn(run, previous)[j].summary) : [];
  }

  // The lines into a node as its call starts: a flow from its document or, for a parse, the structure; dotted, the
  // call's own document, read again; faint, the other outputs it was shown.
  /** @param {RunTab} tab @param {string} id @returns {Line[]} */
  function linesInto(tab, id) {
    const [kind, i] = splitId(id);
    if (kind === roundKind(1)) return [[docId(tab.run.docs[i]), id, 'flow', 'draw']];
    const own = ownDocument(tab.run, id);
    /** @type {Line[]} */
    const lines = kind === 'parsed' ? [['schema', id, 'flow', 'draw']] : [];
    if (own) lines.push([own, id, 'self', 'fade']);
    return [...lines, ...shownInputs(tab, id).map((from) => /** @type {Line} */ ([from, id, 'context']))];
  }

  /** @param {RunTab} tab @param {string} to @returns {Line[]} */
  const citeLines = (tab, to) => cites(tab.run, to).map((from) => [from, to, 'cite', 'draw']);

  /** @param {Tab} tab @param {Line[]} lines */
  function drawLines(tab, lines) {
    for (const [from, to, kind, how] of lines) addEdge(tab, from, to, kind, how);
  }

  /** @param {RunTab} tab @param {string} to */
  const showCites = (tab, to) => drawLines(tab, citeLines(tab, to));

  // ---------- drawing a run ----------
  // A replay's timeline and a live run's events both draw through these, so the two look the same.

  /** @param {Tab} tab @param {NodeKind} kind */
  const hasNodes = (tab, kind) => [...tab.nodes.values()].some((n) => n.kind === kind);
  /** @param {Tab} tab @param {NodeKind} kind */
  const settledCount = (tab, kind) =>
    tab.fold ? (tab.fold.settled.get(kind) ?? 0) : [...tab.nodes.values()].filter((n) => n.kind === kind && !n.el.classList.contains('working')).length;

  /** @param {StageKind} kind */
  const phaseOf = (kind) => (isRound(kind) ? roundName(kind) : STAGE_PHASE[kind]);

  /** @param {RunTab} tab @param {StageKind} kind */
  function showProgress(tab, kind) {
    const progress = kind === 'schema' || kind === 'answer' ? '' : ` · ${settledCount(tab, kind)} of ${tab.run.docs.length}`;
    tabStatus(tab, `running · ${phaseOf(kind)}${progress}`);
  }

  /** @param {Run} run @param {StageKind} kind */
  function stepLine(run, kind) {
    if (isRound(kind)) return `${roundName(kind)} · extracting relevance`;
    return kind === 'parsed' ? `parsing ${plural(run.docs.length, 'document')}` : phaseOf(kind);
  }

  /** @param {RunTab} tab @param {Narrator} narr @param {StageKind} kind */
  function beginStage(tab, narr, kind) {
    narr.log('main', stepLine(tab.run, kind));
    narr.stage = isRound(kind) ? 'relevance' : STAGE_OF[kind];
    showProgress(tab, kind);
  }

  /** @param {RunTab} tab @param {string} id */
  function appear(tab, id) {
    addNode(tab, id);
    drawLines(tab, linesInto(tab, id));
  }

  /** @param {Run} run @param {string} id */
  function isVoid(run, id) {
    const [kind, i] = splitId(id);
    if (isRound(kind)) return !summariesIn(run, kind)[i].summary;
    return kind === 'parsed' && !hasRecords(run.parsed[i]);
  }

  /** @param {Run} run @param {string} id */
  function settledLine(run, id) {
    const [kind, i] = splitId(id);
    if (isRound(kind)) return `${docLabel(run, i)} — ${summariesIn(run, kind)[i].summary ? 'snippet ready' : 'nothing relevant'}`;
    if (kind === 'schema') return schemaClasses(/** @type {Schema} */ (run.schema).code).join(', ') || 'structure ready';
    return `${docLabel(run, i)} — ${hasRecords(run.parsed[i]) ? 'structure ready' : 'nothing parsed'}`;
  }

  // Draws a node's call as finished; again, for a document read in parts, as each later part adds to it.
  /** @param {RunTab} tab @param {Narrator} narr @param {string} id */
  function settle(tab, narr, id) {
    const { run } = tab;
    const [kind] = /** @type {[StageKind, number | null]} */ (splitId(id));
    settleNode(tab, id, { empty: isVoid(run, id) });
    showCites(tab, id);
    narr.logAs(id, 'sub', settledLine(run, id), { node: id });
    showProgress(tab, kind);
  }

  // An answer's node settles, with solid lines from what it cites.
  /** @param {RunTab} tab @param {string} id */
  function drawAnswer(tab, id) {
    settleNode(tab, id);
    if (tab.fold) unfoldTrail(tab, id);
    showCites(tab, id);
  }

  /** @param {RunTab} tab @param {Narrator} narr */
  function showAnswer(tab, narr, id = RUN_ANSWER) {
    drawAnswer(tab, id);
    const answer = answerOf(tab.run, id);
    const flag = answer.simulated ? narr.say('main', h('span', { class: 'flag' }, SIMULATED_LABEL)) : null;
    const said = narr.say('answer', answer.text, { node: id });
    tab.chatAnswer = flag ?? said;
    const sentences = sentenceList(tab, id);
    if (sentences) said.after(sentences);
    updateComposer();
  }

  // Under an answer, folded until opened: the sentences its trail quotes, by document; each opens its document's popup there.
  /** @param {RunTab} tab @param {string} id */
  function sentenceList(tab, id) {
    const trail = answerTrail(tab, id);
    const line = trail && sentencesFrom(trail);
    if (!trail || !line) return null;
    /** @param {LoadedFile} file @param {TextRange[]} sentences */
    const group = (file, sentences) =>
      h(
        'div',
        { class: 'quoted' },
        h('button', { class: 'ref quoted-name', type: 'button', onclick: () => jumpTo(docId(file)) }, file.name),
        sentences.map((at) => sentenceButton(file, at, 'sentence', h('mark', {}, file.text.slice(at.start, at.end)))),
      );
    const groups = tab.run.docs.map((file, d) => (trail.quoted[d].sentences.length ? group(file, trail.quoted[d].sentences) : null));
    return fold(line, groups, 'fold sentences');
  }

  /** @param {RunTab} tab @param {Narrator} narr */
  function finishRun(tab, narr) {
    const { run } = tab;
    sayReady(narr, run.answers[0].model, run.simulated, run.totals);
    tabStatus(tab, ...doneLine(run));
    rerenderIfOpen(tab, RUN_ANSWER);
  }

  // Marks every stage node still working as stopped, while a document being read just settles; with no
  // answer, the next question starts a new run.
  /** @param {Tab} tab @param {Narrator} narr @param {Ending} ending @param {string} [status] */
  function failRun(tab, narr, ending, status = [ending.outcome, ending.error].filter(Boolean).join(' · ')) {
    for (const node of tab.nodes.values()) {
      if (node.el.classList.contains('working')) settleNode(tab, node.id, { failed: node.kind !== 'doc' });
    }
    if (tab.fold) foldsStopped(tab);
    sayEnding(narr, ending);
    tabStatus(tab, ...errorLine(status));
  }

  /** @param {RunTab} tab @param {Narrator} narr */
  function buildTimeline(tab, narr) {
    const { run } = tab;
    /** @type {{ at: number, run: () => void }[]} */
    const steps = [];
    /** @param {number} t @param {() => void} fn */
    const at = (t, fn) => steps.push({ at: t, run: fn });
    const n = run.docs.length;
    const { calls } = run;
    // A recorded run's nodes settle in the order its calls finished; a simulated run's, in document order.
    /** @param {PerDocumentKind} kind */
    const finishOrder = (kind) => (calls ? range(n).sort((a, b) => calls[`${kind}-${a}`].step - calls[`${kind}-${b}`].step) : range(n));
    /** @param {number} t0 @param {PerDocumentKind} kind */
    const perDocument = (t0, kind) => stage(at, t0, finishOrder(kind), n, (i) => appear(tab, `${kind}-${i}`), (i) => settle(tab, narr, `${kind}-${i}`));
    let t = TIMING.start;

    at(t, () => {
      narr.stage = READING;
      narr.log('main', `reading ${plural(n, 'document')}`);
      if (run.simulated) {
        narr.say('sub', h('span', { class: 'flag' }, `simulated · each relevance snippet is the file's first ${SUMMARY_CHARS} characters, and the citations are made up`));
      }
      tabStatus(tab, `running · reading · 0 of ${n}`);
    });
    /** @param {number} i */
    const docAt = (i) => docId(run.docs[i]);
    t = stage(at, t, range(n), n, (i) => markWorking(tab, docAt(i)), (i, rank) => {
      settleNode(tab, docAt(i));
      tabStatus(tab, `running · reading · ${rank + 1} of ${n}`);
    }, PACE.read);
    at(t, () => {
      narr.log('sub', `read ${plural(n, 'file')} · ${fmtBytes(textBytes(run.docs))}`);
      narr.stage = null;
    });

    for (const kind of roundKinds(run.rounds.length)) {
      t += TIMING.phaseGap;
      at(t, () => beginStage(tab, narr, kind));
      t = perDocument(t, kind);
    }

    t += TIMING.phaseGap;
    at(t, () => {
      beginStage(tab, narr, 'schema');
      appear(tab, 'schema');
    });
    t += TIMING.schemaWork;
    at(t, () => settle(tab, narr, 'schema'));

    t += TIMING.phaseGap;
    at(t, () => beginStage(tab, narr, 'parsed'));
    t = perDocument(t, 'parsed');

    t += TIMING.phaseGap;
    at(t, () => {
      beginStage(tab, narr, 'answer');
      appear(tab, RUN_ANSWER);
    });
    t += TIMING.answerWork;
    at(t, () => {
      showAnswer(tab, narr);
      finishRun(tab, narr);
      for (let k = 1; k < run.answers.length; k += 1) redrawAsk(tab, answerId(k));
      if (!run.simulated) offerRecordedAsks(narr);
    });
    t += TIMING.litDelay;
    at(t, () => lightRun(tab));
    return steps;
  }

  /** @param {RunTab} tab @param {Narrator} narr */
  function playTimeline(tab, narr) {
    const steps = buildTimeline(tab, narr);
    if (reduceMotion()) {
      steps.sort((a, b) => a.at - b.at).forEach((s) => s.run());
      return;
    }
    for (const s of steps) tab.timers.push(setTimeout(s.run, s.at));
  }

  // ---------- live ----------

  // Fills in the tab's run from one of its events and draws it, in that tab whether or not it is shown.
  /** @param {RunTab} tab @param {Narrator} narr @param {ServerEvent} event */
  function drawEvent(tab, narr, event) {
    const { run } = tab;
    // An event names a document of its own run.
    /** @param {string} id */
    const doc = (id) => /** @type {number} */ (docIndex(run, id));
    switch (event.type) {
      case 'run_started':
        Object.assign(run, withIds(event.document_ids));
        if (event.rounds !== run.rounds.length) makeRoomForRounds(tab, event.rounds);
        narr.log('main', `sent ${plural(run.docs.length, 'document')} to R3Con · ${fmtBytes(textBytes(run.docs))}`);
        break;
      case 'relevance_started':
        if (!hasNodes(tab, roundKind(event.round))) beginStage(tab, narr, roundKind(event.round));
        appear(tab, `${roundKind(event.round)}-${doc(event.doc_id)}`);
        break;
      case 'relevance_finished':
        addSnippet(tab, narr, doc(event.doc_id), event);
        break;
      case 'schema_started':
        beginStage(tab, narr, 'schema');
        appear(tab, 'schema');
        break;
      case 'reasoning_started':
        beginStage(tab, narr, 'answer');
        appear(tab, RUN_ANSWER);
        break;
      case 'schema_finished':
        run.schema = { code: event.schema_code, thought: event.thought };
        settle(tab, narr, 'schema');
        break;
      case 'parsing_started':
        if (!hasNodes(tab, 'parsed')) beginStage(tab, narr, 'parsed');
        appear(tab, `parsed-${doc(event.doc_id)}`);
        break;
      case 'parsing_finished':
        run.parsed[doc(event.doc_id)] = { parse: event.parse, provenance: docIndices(run, event.provenance) };
        settle(tab, narr, `parsed-${doc(event.doc_id)}`);
        break;
      case 'reasoning_finished':
        Object.assign(run.answers[0], { text: event.answer, provenance: provenanceOf(run, event.provenance) });
        showAnswer(tab, narr);
        break;
      case 'run_finished':
        run.totals = totalsOf(event);
        finishRun(tab, narr);
        lightRun(tab);
        break;
      case 'run_failed':
        failRun(tab, narr, endingOf(stopping(tab), event));
        break;
    }
  }

  // A document read in parts finishes once per part: each adds to its node, drawn again from all of them. A part's
  // quotes are placed in the whole document, so they add up as they are, each with its part's trail node.
  /** @param {RunTab} tab @param {Narrator} narr @param {number} i @param {Extract<ServerEvent, { type: 'relevance_finished' }>} event */
  function addSnippet(tab, narr, i, { round, doc_id, part, summary, provenance, quotes }) {
    const { run } = tab;
    const summaries = run.rounds[round - 1];
    const node = summaryNode(round, part == null ? doc_id : `${doc_id}.${part}`);
    const earlier = summaries[i] ?? { summary: '', provenance: [], quotes: [] };
    summaries[i] = {
      summary: [earlier.summary, summary].filter(Boolean).join('\n\n'),
      provenance: [...new Set([...earlier.provenance, ...docIndices(run, provenance)])],
      quotes: [...earlier.quotes, ...quotesInUnits(run.docs[i].text, quotes).map((q) => ({ ...q, summary: node }))],
    };
    settle(tab, narr, `${roundKind(round)}-${i}`);
  }

  // Should a live run's first event name other rounds than it asked for, it is planned again before any is drawn.
  /** @param {RunTab} tab @param {number} rounds */
  function makeRoomForRounds(tab, rounds) {
    tab.run.rounds = emptyRounds(rounds);
    tab.layout = plan(tab, nodeOrigin(tab));
    if (tab === state.shown) fitShown(true);
  }

  /** @param {RunTab} tab @param {Narrator} narr */
  async function streamRun(tab, narr) {
    const { run } = tab;
    const stream = openStream('run', tab, run);
    try {
      const { question, model, endpoint } = run.answers[0];
      const body = { question, model, endpoint, documents: run.docs.map((f) => f.text), names: run.docs.map((f) => f.name), relevance_rounds: run.rounds.length };
      const started = await api('/api/runs', { method: 'POST', body });
      jobStarted(stream, (await started.json()).run_id);
      await follow(`/api/runs/${run.id}/events`, (event) => drawEvent(tab, narr, event));
    } catch (error) {
      const said = explain(error, /** @type {Error} */ (error).message, 'the live run stopped');
      failRun(tab, narr, { outcome: inStage('failed', narr.stage), error: said }, said);
    } finally {
      closeStream(stream);
    }
  }

  // An ask runs only reasoning, over what the run built; its node waits in the answer column until it lands.
  /** @param {RunTab} tab @param {string} question @param {string} model @param {string | null} endpoint */
  async function askOver(tab, question, model, endpoint) {
    const { run } = tab;
    /** @type {Answer} */
    const ask = { question, model, source: 'live', text: null, provenance: null, totals: null };
    const id = addAsk(tab, ask);
    const narr = askTurn(tab, question, model);
    const stream = openStream('ask', tab, ask);
    // An ask leaves the status alone, but its stop was asked for there: once that has ended, the status
    // says so until it next changes, and the tab keeps its run's own line.
    /** @type {string | null} */
    let stopped = null;
    /** @param {Ending} ending */
    const failed = (ending) => {
      if (!landed(tab, id)) settleNode(tab, id, { failed: true });
      sayEnding(narr, ending);
    };
    /** @param {ServerEvent} event */
    const onEvent = (event) => {
      if (event.type === 'reasoning_finished') {
        Object.assign(ask, { text: event.answer, provenance: provenanceOf(run, event.provenance) });
        showAnswer(tab, narr, id);
        lightLanded(tab, id);
      }
      if (event.type === 'ask_finished') {
        ask.totals = totalsOf(event);
        sayReady(narr, model, false, ask.totals);
        rerenderIfOpen(tab, id);
      }
      if (event.type === 'ask_failed') {
        const ending = endingOf(stream.stopping, event);
        failed(ending);
        if (stream.stopping) stopped = `ask ${ending.outcome}`;
      }
    };
    try {
      const asked = await api(`/api/runs/${run.id}/asks`, { method: 'POST', body: { question, model, endpoint } });
      jobStarted(stream, (await asked.json()).ask_id);
      await follow(`/api/asks/${ask.id}/events`, onEvent);
    } catch (error) {
      failed({ outcome: 'failed', error: explain(error, /** @type {Error} */ (error).message, 'the ask stopped') });
    } finally {
      closeStream(stream);
      if (stopped && tab === state.shown && !stopping(tab)) errorStatus(stopped);
    }
  }

  /** @param {Tab} tab */
  const docPositions = (tab) => new Map([...tab.nodes.values()].filter((n) => n.kind === 'doc').map((n) => [n.id, { x: n.x, y: n.y }]));

  // Each call is shown about one input per row, so faint lines grow as the rows squared; fading them keeps the grey
  // steady.
  /** @param {Tab} tab @param {number} rows */
  function fadeFaint(tab, rows) {
    tab.drawing.style.setProperty('--context-fade', String(Math.min(1, FAINT_FADE_FROM_DOCS / rows)));
  }

  // Plays the shown tab's run, its document nodes starting from `at`.
  /** @param {RunTab} tab */
  function playRun(tab, at = docPositions(tab)) {
    cancelTimers(tab);
    clearPanel(tab);
    fadeFaint(tab, tab.run.docs.length);
    tab.turn.replaceChildren();
    tab.userMoved = false;
    const narr = (tab.narr = narrator(tab, tab.turn, 'starting'));
    tabStatus(tab, 'running · starting');
    // A folded tab's rows carry over into its run, each drawn where it was, so that starting doesn't jump.
    startFolding(tab);
    if (tab.fold) refold(tab, { was: at, fit: false });
    syncDocNodes(tab, { at, animate: true, replan: true });
    mode.play(tab, narr);
    updateComposer();
  }

  /** @param {string} question */
  function startRun(question) {
    const from = state.shown;
    const at = docPositions(from);
    const opened = from.run ? copyTab(from) : from;
    const tab = Object.assign(opened, { run: { ...mode.newRun(question, opened.files), lit: null }, turn: addTurn(opened, question) });
    if (tab === from) renderFiles();
    else showTabContents(tab);
    playRun(tab, at);
    revealTab(tab);
  }

  // ---------- asks ----------
  // An ask answers again over the run's representation; its answer node joins the run's in the answer column.

  /** @param {Tab} tab @param {string} question @param {string} model */
  function askTurn(tab, question, model) {
    const narr = narrator(tab, addTurn(tab, question));
    narr.log('main', `asking · ${model}`);
    return narr;
  }

  /** @param {RunTab} tab @param {Answer} ask */
  function addAsk(tab, ask) {
    const id = answerId(tab.run.answers.push(ask) - 1);
    tab.layout.set(id, freeAnswerSpot(tab));
    addNode(tab, id);
    fitShown(true);
    return id;
  }

  // Draws a landed ask again, as a replay does, without saying it in the chat again.
  /** @param {RunTab} tab @param {string} id */
  function redrawAsk(tab, id) {
    addNode(tab, id);
    drawAnswer(tab, id);
  }

  // Shifted per ask, so that each simulated answer cites other documents than the last.
  /** @param {Run} run @param {string} question @param {string} model @returns {Answer} */
  function simulatedAsk(run, question, model) {
    return { question, model, source: 'simulated', ...madeUpAnswer(run, question, run.answers.length), totals: null };
  }

  // A recorded answer when one has this question and model over the recorded run, else a simulated one.
  /** @param {RunTab} tab @param {string} question @param {string} model */
  function prototypeAsk(tab, question, model) {
    const { run } = tab;
    const recorded = !run.simulated && RECORDED_ASKS.find((entry) => entry.question === question && entry.model === model);
    /** @type {Answer} */
    const ask = recorded ? { ...recorded, provenance: provenanceOf(run, recorded.provenance), source: 'recorded' } : simulatedAsk(run, question, model);
    const id = addAsk(tab, ask);
    const narr = askTurn(tab, question, model);
    showAnswer(tab, narr, id);
    sayReady(narr, model, ask.source === 'simulated', ask.totals);
    lightLanded(tab, id);
  }

  // After the recorded replay, each recorded ask is a click away: it fills the question and picks its model.
  /** @param {Narrator} narr */
  function offerRecordedAsks(narr) {
    if (!RECORDED_ASKS.length) return;
    /** @param {RecordedAsk} ask @param {number} i */
    const offer = (ask, i) => h('button', { class: 'ref', type: 'button', 'data-offer': i }, `${ask.question} · ${bareModel(ask.model)}`);
    narr.say('main', 'recorded asks over this run · any other question or model is simulated', { dim: true });
    RECORDED_ASKS.forEach((ask, i) => narr.say('sub', offer(ask, i)));
  }

  /** @param {{ question: string, model: string }} ask */
  function prepareAsk({ question, model }) {
    const listed = [...modelPick.options].some((option) => option.value === model);
    modelPick.value = listed ? model : '';
    if (!listed) modelOther.value = model;
    onPick();
    questionEl.value = question;
    questionEl.focus();
  }

  // ---------- tabs ----------

  /** @param {Tab} tab */
  function tabLabel(tab) {
    if (tab.run) return answerLabel(tab.run.answers[0]);
    return tab.files.length ? `new · ${plural(tab.files.length, 'file')}` : 'new';
  }

  // A tab can't close while its run or an ask over it is going.
  /** @param {Tab} tab */
  const tabBusy = (tab) => mode.runGoing(tab) || streaming('ask', tab);

  // Lists the tab.
  /** @param {Tab} tab */
  function openTab(tab) {
    state.tabs.push(tab);
    tab.button.append(tabLabel(tab));
    $('run-tabs').append(tab.button, tab.close);
    $('ask-form').before(tab.chat);
    if (tab.files.length) tellChat(tab, `Loaded ${fileCount(tab.files)}.`);
    return tab;
  }

  /** @param {Tab} from */
  function copyTab(from) {
    return openTab(newTab(from.files, from.view));
  }

  function updateTabs() {
    for (const tab of state.tabs) {
      const shown = tab === state.shown;
      const label = tabLabel(tab);
      const labelNode = /** @type {ChildNode} */ (tab.button.lastChild);
      // A label grows as files are added, which can push the shown tab out of the row's view.
      if (labelNode.textContent !== label) {
        labelNode.textContent = label;
        if (shown) revealTab(tab);
      }
      tab.button.setAttribute('aria-selected', String(shown));
      tab.button.tabIndex = shown ? 0 : -1;
      tab.close.hidden = state.tabs.length === 1;
      tab.close.disabled = tabBusy(tab);
      tab.button.ariaKeyShortcuts = tab.close.hidden || tab.close.disabled ? null : 'Delete';
    }
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', state.shown.button.id);
  }

  /** @param {Tab} tab */
  function showTabContents(tab) {
    finishSlide(state.shown);
    // A drawing taken out of the page mid-animation never hears the animation end.
    for (const el of state.shown.drawing.querySelectorAll('.enter, .draw, .fade')) endAnimation(el);
    closeInspector();
    endGesture();
    state.hovered = null;
    state.shown.chat.hidden = true;
    state.shown = tab;
    world.replaceChildren(tab.drawing);
    tab.chat.hidden = false;
    revealChat(tab);
    applyView();
    refreshHighlight();
    closeFile();
    renderFiles();
    showFilesLines(false);
  }

  // On a phone the tab row scrolls, and the shown tab could be out of sight.
  /** @param {Tab} tab */
  function revealTab(tab) {
    for (const el of [tab.button, tab.close]) if (!el.hidden) el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  /** @param {Tab} tab @param {Map<string, Point>} [at] where to draw document nodes the tab hasn't drawn yet */
  function switchTab(tab, at) {
    if (tab === state.shown) return;
    // Each tab keeps its question. Once shown its draft is emptied: new run leaves a tab without saving the box.
    state.shown.draft = questionEl.value;
    questionEl.value = tab.draft;
    tab.draft = '';
    showTabContents(tab);
    // Nodes drawn while their tab was hidden had no size to fit by.
    syncDocNodes(tab, { at });
    showStatus(tab);
    updateComposer();
    revealTab(tab);
  }

  /** @param {Tab} tab */
  function closeTab(tab) {
    if (state.tabs.length === 1) return;
    // Only Delete gets here for a busy tab: its × is disabled, and hidden from assistive tech.
    if (tabBusy(tab)) return setStatus(tabStillGoing(tab));
    cancelTimers(tab);
    const i = state.tabs.indexOf(tab);
    state.tabs.splice(i, 1);
    tab.button.remove();
    tab.close.remove();
    tab.chat.remove();
    if (tab === state.shown) switchTab(state.tabs[Math.min(i, state.tabs.length - 1)]);
    else updateTabs();
    state.shown.button.focus();
  }

  /** @type {Record<string, number>} */
  const TAB_KEYS = { ArrowRight: 1, ArrowLeft: -1 };

  /** @param {KeyboardEvent} e */
  function onTabKey(e) {
    const tab = state.tabs.find((t) => t.button === e.target);
    if (!tab) return;
    if (e.key === 'Delete') return closeTab(tab);
    const step = TAB_KEYS[e.key];
    if (!step) return;
    e.preventDefault();
    const next = state.tabs[(state.tabs.indexOf(tab) + step + state.tabs.length) % state.tabs.length];
    switchTab(next);
    next.button.focus();
  }

  // ---------- files ----------

  /** @param {'chat' | 'files'} name */
  function showTab(name) {
    for (const tab of ['chat', 'files']) {
      $(`tab-${tab}`).setAttribute('aria-selected', String(tab === name));
      $(`${tab}-pane`).hidden = tab !== name;
    }
    revealChat(state.shown);
  }

  function renderFiles() {
    const { files, run } = state.shown;
    const sampleOnly = files.length > 0 && files.every((f) => f.sample);
    const head = `${fileCount()} · ${fmtBytes(totalBytes(files))}${sampleOnly ? ' · sample memos' : ''}`;
    $('files-head').textContent = files.length ? head : NO_FILES;
    $('file-list').replaceChildren(
      ...files.map((f) =>
        h(
          'li',
          {},
          h('button', { class: 'file', type: 'button', onclick: () => openFile(f) }, h('span', { class: 'name' }, f.name), h('span', { class: 'used' }), h('span', { class: 'size' }, fmtBytes(f.size))),
          run ? null : h('button', { class: 'btn remove', type: 'button', 'aria-label': `remove ${f.name}`, onclick: () => removeFile(f) }, 'remove'),
        ),
      ),
    );
    $('files-count').textContent = String(files.length);
    $('files-locked').hidden = !run;
    $('file-actions').hidden = $('drop-hint').hidden = Boolean(run);
    $('drop-overlay').textContent = run ? DROP_LOCKED : DROP_ADDS;
    markFiles();
  }

  // In place, so that a focused file stays focused: how many sentences the lit answer's trail quotes of each file, and
  // the files off it dimmed.
  function markFiles() {
    const { files, trail } = state.shown;
    $('file-list').querySelectorAll('.file').forEach((button, d) => {
      const sentences = trail?.quoted[d].sentences.length ?? 0;
      const named = plural(sentences, 'highlighted sentence');
      button.classList.toggle('off', Boolean(trail && !trail.nodes.has(docId(files[d]))));
      fill(/** @type {Element} */ (button.querySelector('.used')), sentences ? [h('span', { 'aria-hidden': 'true', title: named }, `${sentences} ¶`), h('span', { class: 'vh' }, named)] : null);
    });
  }

  // Lines restored on switching to their tab were said already, so they are written with the live regions off.
  function showFilesLines(announce = true) {
    const { filesSaid, filesError } = state.shown;
    const said = $('files-said');
    const error = $('files-error');
    said.ariaLive = announce ? 'polite' : 'off';
    error.ariaLive = announce ? 'assertive' : 'off';
    // Shown before it is filled, so that it is announced.
    error.hidden = !filesError;
    // A line written again unchanged would be announced again.
    if (error.textContent !== filesError) error.textContent = filesError;
    if (said.textContent !== filesSaid) said.textContent = filesSaid;
  }

  /** @param {LoadedFile} file */
  function openFile(file) {
    showTab('files');
    // A folded document is shown by its fold until it is shown in the panel.
    const shown = state.shown.nodes.has(shownAs(state.shown, docId(file)));
    fill(
      $('file-view'),
      h('button', { class: 'btn back', type: 'button', id: 'file-back', onclick: closeFile }, '‹ all files'),
      h('div', { class: 'insp-title' }, h('span', { class: 'g' }, '● '), file.name),
      markedFile(state.shown, file),
      shown ? h('button', { class: 'btn back', type: 'button', onclick: () => jumpTo(docId(file)) }, 'show in panel ›') : null,
    );
    $('file-list-view').hidden = true;
    $('file-view').hidden = false;
    $('file-view').scrollTop = 0;
  }

  function closeFile() {
    $('file-view').hidden = true;
    $('file-list-view').hidden = false;
  }

  /** @param {File} file @returns {Promise<string>} */
  function readText(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => resolve('');
      reader.readAsText(file);
    });
  }

  // No text holds a NUL byte but UTF-16, which readText decodes after a byte-order mark, as Windows writes it;
  // even then, no code unit of text is zero, as one of UTF-32 or of binary that starts the same way is.
  /** @param {Uint8Array} start */
  function looksLikeText(start) {
    if (!UTF16_BOMS.some(([first, second]) => start[0] === first && start[1] === second)) return !start.includes(0);
    for (let i = 2; i + 1 < start.length; i += 2) if (!start[i] && !start[i + 1]) return false;
    return true;
  }

  /** @param {File} file @returns {Promise<FileRead>} */
  async function readFile(file) {
    if (hasSuffix(file, PDF_SUFFIX)) return mode.extract(file, NO_TEXT_LAYER);
    if (EXTRACTED_SUFFIXES.some((suffix) => hasSuffix(file, suffix))) return mode.extract(file, NO_TEXT);
    // A file that can't be read goes on to readText, which reads it as no text.
    const start = new Uint8Array(await file.slice(0, SNIFF_BYTES).arrayBuffer().catch(() => new ArrayBuffer(0)));
    if (looksLikeText(start)) return { text: await readText(file) };
    return { why: hasSuffix(file, OLD_WORD_SUFFIX) ? OLD_WORD : NOT_TEXT };
  }

  /** @param {File} file @param {FileRead} read @returns {Unpacked} */
  const oneDocument = (file, read) => ({ docs: [{ name: file.name, size: file.size, ...read }], skipped: 0 });

  /** @param {File} file */
  async function readDocuments(file) {
    return hasSuffix(file, ZIP_SUFFIX) ? mode.unzip(file) : oneDocument(file, await readFile(file));
  }

  // A tab's batches are read and added one after another, so that each one's lines in the files pane stand.
  /** @param {Picked} batch @param {Tab} [tab] */
  function addFiles(batch, tab = state.shown) {
    tellFiles(tab, { filesError: '' });
    tab.reading += 1;
    updateHint();
    tab.adding = tab.adding
      .then(() => addBatch(batch, tab))
      .catch((error) => failedToAdd(tab, error))
      .finally(() => {
        tab.reading -= 1;
        updateHint();
      });
  }

  /** @param {Tab} tab @param {{ filesSaid?: string, filesError?: string }} lines */
  function tellFiles(tab, lines) {
    Object.assign(tab, lines);
    showFilesLines();
  }

  // A user's action on a tab's files clears its error line, and what its batches say later only adds to it.
  /** @param {Tab} tab @param {string} text */
  function addFilesError(tab, text) {
    tellFiles(tab, { filesError: [tab.filesError, text].filter(Boolean).join(' ') });
  }

  /** @param {string[]} names */
  function someNames(names) {
    const more = names.length - NAMED_AT_MOST;
    return more > 0 ? `${names.slice(0, NAMED_AT_MOST).join(', ')} and ${more} more` : names.join(', ');
  }

  /** @param {string[]} names @param {string} why */
  function notAdded(names, why) {
    return names.length ? `Not added: ${someNames(names)}. ${why}` : '';
  }

  // By path, a part at a time and by character code, as r3con's walk and r3con app's zips order files: a/b.txt before a-b/x.txt.
  /** @param {File} a @param {File} b */
  function byPath(a, b) {
    const [x, y] = [a.name.split('/'), b.name.split('/')];
    const i = x.findIndex((part, j) => part !== y[j]);
    if (i < 0) return x.length - y.length;
    return i < y.length && x[i] < y[i] ? -1 : 1;
  }

  /** @param {File[]} files */
  const pickFiles = (files) => files.filter((f) => !f.name.startsWith('.') && !f.name.startsWith(LOCK_FILE_PREFIX)).sort(byPath);

  /** @param {Tab} tab @param {unknown} error */
  function failedToAdd(tab, error) {
    tellFiles(tab, { filesSaid: '' });
    addFilesError(tab, explain(error, "Couldn't add those files.", "couldn't add files"));
  }

  /** @param {Picked} batch @param {Tab} tab */
  async function addBatch({ files, skipped, failed }, tab) {
    // A tab closed, or run, since its files were picked takes none of them, so they aren't read.
    if (!state.tabs.includes(tab)) return;
    const picked = pickFiles(files);
    if (tab.run) return addFilesError(tab, notAdded([...failed, ...picked.map((f) => f.name)], RAN_FIRST) || RAN_FIRST);
    if (!picked.length && !failed.length) return addFilesError(tab, skipped ? noDocuments(skipped) : 'No files found there. Try picking the files themselves.');
    /** @type {LoadedFile[]} */
    const read = [];
    /** @type {Map<string, string[]>} */
    const refused = new Map(failed.length ? [[BROWSER_COULD_NOT_READ, failed]] : []);
    /** @param {string} why @param {string} name */
    const refuse = (why, name) => refused.set(why, [...(refused.get(why) ?? []), name]);
    /** @param {DocumentRead} r */
    const take = (r) => {
      if ('why' in r) refuse(r.why, r.name);
      else if (tab.run) refuse(RAN_FIRST, r.name);
      else read.push(loaded({ name: r.name, size: r.size, text: r.text, sample: false }));
    };
    if (picked.length) tellFiles(tab, { filesSaid: `Reading ${plural(picked.length, 'file')}…` });
    const unpacked = await Promise.all(picked.map(readDocuments));
    if (!state.tabs.includes(tab)) return;
    unpacked.flatMap((u) => u.docs).forEach(take);
    skipped += unpacked.reduce((sum, u) => sum + u.skipped, 0);
    for (const [why, names] of refused) addFilesError(tab, notAdded(names, why));
    if (!read.length) {
      tellFiles(tab, { filesSaid: '' });
      return;
    }
    // Taken now, as files can be removed while a batch is read; files added over the sample memos take their place.
    const left = tab.files.filter((f) => !f.sample);
    const replaced = tab.files.length - left.length;
    tab.files = [...left, ...read];
    const added = `${plural(read.length, 'file')} added`;
    const said = replaced ? `${added}, in place of the ${plural(replaced, 'sample memo')}` : added;
    filesChanged(tab, skipped ? `${said}; ${skipped} skipped` : said);
  }

  // The re-render drops the clicked button, and the focus with it.
  /** @param {LoadedFile} file @param {{ refocus?: boolean }} [options] */
  function removeFile(file, { refocus = true } = {}) {
    const { files } = state.shown;
    const i = files.indexOf(file);
    changeFiles(files.filter((f) => f !== file), `Removed ${file.name}`);
    if (!refocus) return;
    /** @type {NodeListOf<HTMLButtonElement>} */
    const buttons = $('file-list').querySelectorAll('.remove');
    (buttons[Math.min(i, buttons.length - 1)] ?? $('file-input')).focus();
  }

  let lastPress = -Infinity;

  function removeAllOrUndo() {
    const tooSoon = performance.now() - lastPress < PRESS_GUARD_MS;
    lastPress = performance.now();
    if (tooSoon) return;
    const tab = state.shown;
    const { files, removed } = tab;
    if (removed) changeFiles(removed.files, `Restored ${fileCount(removed.files)}`, { at: removed.at });
    else changeFiles([], `Removed ${fileCount()}`, { removed: { files, at: docPositions(tab) } });
  }

  /** @param {LoadedFile[]} files @param {string} what @param {FilesChange} [change] */
  function changeFiles(files, what, change) {
    const tab = state.shown;
    tab.files = files;
    tellFiles(tab, { filesError: '' });
    filesChanged(tab, what, change);
  }

  // Undo stays only until the tab's files change some other way.
  /** @param {Tab} tab @param {string} what @param {FilesChange} [change] */
  function filesChanged(tab, what, { removed, at } = {}) {
    tab.removed = removed ?? null;
    // Rendered again for another tab's files, the pane would close the shown tab's open file, and drop the focus.
    if (tab === state.shown) {
      closeFile();
      renderFiles();
    }
    updateComposer();
    syncDocNodes(tab, { at });
    readyStatus();
    tellFiles(tab, { filesSaid: `${what}.` });
    tellChat(tab, `${what}: the next question runs over ${fileCount(tab.files)}.`);
  }

  // How a file in a dropped or picked folder is taken, by its path below that folder, mirroring r3con's folder walk:
  // by suffix, with no Office lock file and nothing hidden or under __MACOSX. Only a file left out by its suffix is
  // counted as skipped.
  /** @param {string[]} parts @param {boolean} isFile @returns {'read' | 'skipped' | 'unsaid'} */
  function folderTakes(parts, isFile) {
    const name = parts.at(-1) ?? '';
    const hidden = parts.some((part) => part.startsWith('.') || part === MACOS_METADATA_FOLDER);
    if (hidden || (isFile && name.startsWith(LOCK_FILE_PREFIX))) return 'unsaid';
    return !isFile || FOLDER_SUFFIXES.some((suffix) => name.toLowerCase().endsWith(suffix)) ? 'read' : 'skipped';
  }

  // A path's parts below the folder that was dropped or picked, the first.
  /** @param {string} path */
  const below = (path) => path.split('/').filter(Boolean).slice(1);
  // A file in a folder is named by its path below it, so that a/notes.txt and b/notes.txt are told apart.
  /** @param {File} file @param {string[]} parts */
  const named = (file, parts) => new File([file], parts.join('/'), { type: file.type, lastModified: file.lastModified });

  // A picked folder's files are taken as a dropped folder's are; picked files, as they are.
  /** @param {File[]} files @returns {Picked} */
  function fromFolders(files) {
    /** @type {Picked} */
    const picked = { files: [], skipped: 0, failed: [] };
    for (const file of files) {
      const parts = below(file.webkitRelativePath);
      const taken = file.webkitRelativePath ? folderTakes(parts, true) : 'read';
      if (taken === 'skipped') picked.skipped += 1;
      if (taken === 'read') picked.files.push(file.webkitRelativePath ? named(file, parts) : file);
    }
    return picked;
  }

  /** @param {FileSystemFileEntry} entry @returns {Promise<File | null>} */
  const fileOf = (entry) => new Promise((resolve) => entry.file(resolve, () => resolve(null)));

  // What is dropped is taken as it is; what is in a dropped folder, as folderTakes says.
  /** @param {DataTransfer} dataTransfer @returns {Promise<Picked>} */
  async function filesFromDrop(dataTransfer) {
    const entries = [...dataTransfer.items].map((item) => item.webkitGetAsEntry?.()).filter((entry) => entry != null);
    if (!entries.length) return { files: [...dataTransfer.files], skipped: 0, failed: [] };
    /** @type {Picked} */
    const dropped = { files: [], skipped: 0, failed: [] };
    /** @param {FileSystemEntry} entry @param {boolean} inFolder */
    const walk = async (entry, inFolder) => {
      const parts = below(entry.fullPath);
      const taken = inFolder ? folderTakes(parts, entry.isFile) : 'read';
      if (taken === 'skipped') dropped.skipped += 1;
      if (taken !== 'read') return;
      if (entry.isFile) {
        const file = await fileOf(/** @type {FileSystemFileEntry} */ (entry));
        if (!file) dropped.failed.push(inFolder ? parts.join('/') : entry.name);
        else dropped.files.push(inFolder ? named(file, parts) : file);
        return;
      }
      const reader = /** @type {FileSystemDirectoryEntry} */ (entry).createReader();
      for (;;) {
        /** @type {FileSystemEntry[]} */
        const batch = await new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
        if (!batch.length) return;
        for (const child of batch) await walk(child, true);
      }
    };
    for (const entry of entries) await walk(entry, false);
    return dropped;
  }

  // ---------- theme ----------

  function currentTheme() {
    return root.dataset.theme;
  }

  function updateThemeLabel() {
    $('theme-btn').textContent = `theme · ${currentTheme()}`;
  }

  function toggleTheme() {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    writeStored(THEME_KEY, next);
    updateThemeLabel();
  }

  function restoreTheme() {
    const saved = readStored(THEME_KEY);
    if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;
    updateThemeLabel();
  }

  // ---------- keys ----------

  // The prototype only notes what r3con app would answer; r3con app keeps the keys themselves.
  /** @type {Map<string, KeyState>} */
  const keyed = new Map();
  /** @type {Map<string, string | null>} */
  const prototypeServers = new Map();

  /** @param {string} key */
  const hintOf = (key) => (key.length >= MIN_HINTED_KEY_CHARS ? key.slice(-KEY_HINT_CHARS) : '');

  /** @returns {KeyStatus} */
  function prototypeStatus() {
    return { ...Object.fromEntries(keyed), other: [...prototypeServers].map(([endpoint, key]) => ({ endpoint, key })) };
  }

  /** @param {string | null} key */
  const keyText = (key) => (key === null ? 'no key' : MASK + key);

  /** @param {string} provider @param {KeyStatus} status */
  function keyState(provider, status) {
    const state = /** @type {KeyState} */ (status[provider]);
    if (!state) return { text: '', removable: false };
    return state.source === 'here' ? { text: keyText(state.hint), removable: true } : { text: `from ${state.variable}`, removable: false };
  }

  /** @param {Server[]} servers */
  function renderServers(servers) {
    /** @param {Server} server */
    const row = ({ endpoint, key }) =>
      h(
        'li',
        { class: 'key-state' },
        h('span', {}, `${endpoint} · ${keyText(key)}`),
        h('button', { class: 'btn remove', type: 'button', 'aria-label': `remove server ${endpoint}`, onclick: () => forget(() => mode.forgetServer(endpoint), SERVER_LABEL, 'endpoint-other') }, 'remove'),
      );
    fill($('server-list'), servers.map(row));
    $('servers-label').hidden = $('server-list').hidden = !servers.length;
  }

  /** @param {KeyStatus} status */
  function renderKeys(status) {
    keysNow = status;
    keysRead = true;
    for (const provider of Object.keys(PROVIDERS)) {
      const { text, removable } = keyState(provider, status);
      $(`state-${provider}`).textContent = text;
      $(`key-state-${provider}`).hidden = !text;
      $(`remove-${provider}`).hidden = !removable;
    }
    renderServers(serversOf(status));
    renderPicker();
    readyStatus();
  }

  async function refreshKeys() {
    renderKeys(await mode.readKeys());
  }

  // ---------- model picker ----------

  // A model on no server is its own id, as before this page knew servers; one on a server, or named like an id, is JSON.
  /** @param {string} model @param {string | null} [endpoint] */
  const modelId = (model, endpoint) => (endpoint || model.startsWith('[') ? JSON.stringify([model, endpoint || null]) : model);
  /** @param {SavedModel} entry */
  const savedId = (entry) => modelId(entry.model, entry.endpoint);

  /** @param {string} id @returns {[string, string | null]} */
  function parseModelId(id) {
    if (!id.startsWith('[')) return [id, null];
    try {
      const [model, endpoint] = JSON.parse(id);
      if (typeof model === 'string' && (endpoint === null || typeof endpoint === 'string')) return [model, endpoint];
    } catch {
      // Not an id this page wrote: read it as a model's name.
    }
    return [id, null];
  }

  function pickedModel() {
    return parseModelId(modelPick.value)[0] || modelOther.value.trim();
  }

  const pickedEndpoint = () => parseModelId(modelPick.value)[1];

  /** @param {string} endpoint */
  function placeOf(endpoint) {
    try {
      const { host, pathname } = new URL(endpoint);
      return { host, path: pathname.replace(/\/$/, '') };
    } catch {
      return { host: endpoint, path: '' };
    }
  }

  // Servers on one host read apart by their paths, and those that still don't by their whole URLs.
  /** @param {string[]} endpoints @returns {Map<string, string>} */
  function serverLabels(endpoints) {
    const hosts = endpoints.map((endpoint) => placeOf(endpoint).host);
    const paths = endpoints.map((endpoint, i) => hosts[i] + placeOf(endpoint).path);
    /** @param {string[]} labels @param {number} i */
    const once = (labels, i) => labels.filter((label) => label === labels[i]).length === 1;
    return new Map(endpoints.map((endpoint, i) => [endpoint, once(hosts, i) ? hosts[i] : once(paths, i) ? paths[i] : endpoint]));
  }

  // Storage is outside input: whatever it holds that isn't a list of named models is left out.
  /** @returns {SavedModel[]} */
  function readSavedModels() {
    try {
      const saved = JSON.parse(readStored(SAVED_MODELS_KEY) ?? '[]');
      /** @param {SavedModel} entry */
      const valid = (entry) => typeof entry?.model === 'string' && entry.model.trim() && ['string', 'undefined'].includes(typeof entry.endpoint);
      return Array.isArray(saved) ? saved.filter(valid) : [];
    } catch {
      return [];
    }
  }

  /** @param {SavedModel[]} saved */
  function writeSavedModels(saved) {
    if (saved.length) writeStored(SAVED_MODELS_KEY, JSON.stringify(saved));
    else removeStored(SAVED_MODELS_KEY);
  }

  /** @type {KeyStatus} */
  let keysNow = {};
  let keysRead = false;
  // The model last picked from the menu, for Escape in the other model field to go back to.
  let menuModel = '';

  // The remembered model stays listed without its key, as after a reload, but never on a server known to be gone.
  function renderPicker() {
    const providers = keyedProviders(keysNow);
    const servers = serversOf(keysNow).map(({ endpoint }) => endpoint);
    const labels = serverLabels(servers);
    const [rememberedModel, rememberedEndpoint] = parseModelId(readStored(MODEL_KEY) || DEFAULT_MODEL);
    const gone = keysRead && !/** @type {(string | null)[]} */ (servers).includes(rememberedEndpoint);
    const remembered = modelId(rememberedModel, gone ? null : rememberedEndpoint);
    /** @param {string} model @param {string | null} [endpoint] @returns {[string, string]} */
    const entry = (model, endpoint, text = model) => [modelId(model, endpoint), text];
    const groups = Object.entries(PROVIDERS)
      .filter(([provider]) => providers.includes(provider))
      .map(([, { label, models }]) => ({ label, entries: models.map((model) => entry(model)) }));
    const offered = new Set(groups.flatMap(({ entries }) => entries.map(([id]) => id)));
    const saved = readSavedModels();
    const savedGroups = [
      { label: SAVED_GROUP, entries: saved.filter((model) => !model.endpoint && !offered.has(savedId(model))).map(({ model }) => entry(model)) },
      ...servers.map((endpoint) => ({
        label: labels.get(endpoint),
        entries: [...saved.filter((model) => model.endpoint === endpoint).map(({ model }) => entry(model, endpoint)), entry('', endpoint, `other model on ${labels.get(endpoint)}…`)],
      })),
    ].filter(({ entries }) => entries.length);
    const allGroups = [...groups, ...savedGroups];
    const listed = new Set(['', ...allGroups.flatMap(({ entries }) => entries.map(([id]) => id))]);
    let pick = modelPick.options.length ? modelPick.value : remembered;
    if (!listed.has(pick)) pick = parseModelId(pick)[0] === '' ? '' : remembered;
    // A model typed on a server and never saved comes back in that server's field, so the menu names its server.
    const [pickModel, pickEndpoint] = parseModelId(pick);
    if (pickModel && pickEndpoint && !listed.has(pick) && listed.has(modelId('', pickEndpoint))) {
      modelOther.value = pickModel;
      pick = modelId('', pickEndpoint);
    }
    const typed = parseModelId(pick)[0] === '' ? modelId(modelOther.value.trim(), parseModelId(pick)[1]) : null;
    /** @param {[string, string]} option */
    const option = ([value, text]) => h('option', { value }, text);
    const unlisted = listed.has(remembered) || remembered === typed ? null : option(entry(...parseModelId(remembered)));
    fill(modelPick, unlisted, allGroups.map(({ label, entries }) => h('optgroup', { label }, entries.map(option))), option(entry('', null, 'other model…')));
    modelPick.value = pick;
    $('model-hint').hidden = !live || providers.length > 0 || servers.length > 0;
    showOtherModel();
  }

  function showOtherModel() {
    const [model, endpoint] = parseModelId(modelPick.value);
    modelOther.hidden = modelOther.disabled = model !== '';
    if (model) menuModel = modelPick.value;
    modelRemove.hidden = !readSavedModels().some((entry) => savedId(entry) === modelPick.value);
    const where = endpoint ? ` on ${serverLabels(serversOf(keysNow).map((server) => server.endpoint)).get(endpoint) ?? endpoint}` : '';
    modelRemove.setAttribute('aria-label', `remove ${model}${where}`);
  }

  function onPick() {
    showOtherModel();
    const model = pickedModel();
    if (model) writeStored(MODEL_KEY, modelId(model, pickedEndpoint()));
    readyStatus();
  }

  function saveTypedModel() {
    const model = modelOther.value.trim();
    if (!model) return;
    const endpoint = pickedEndpoint();
    const id = modelId(model, endpoint);
    const saved = readSavedModels();
    if (!saved.some((entry) => savedId(entry) === id)) writeSavedModels([...saved, endpoint ? { model, endpoint } : { model }]);
    modelOther.value = '';
    renderPicker();
    modelPick.value = id;
    onPick();
    modelPick.focus();
  }

  function cancelTypedModel() {
    modelOther.value = '';
    modelPick.value = menuModel;
    if (modelPick.selectedIndex < 0) modelPick.selectedIndex = 0;
    onPick();
    modelPick.focus();
  }

  function removeSavedModel() {
    const id = modelPick.value;
    writeSavedModels(readSavedModels().filter((entry) => savedId(entry) !== id));
    removeStored(MODEL_KEY);
    renderPicker();
    onPick();
    modelPick.focus();
  }

  // ---------- settings ----------

  /** @param {string} provider @param {Props} props */
  function keyField(provider, props) {
    return h('input', {
      class: 'field key',
      id: `key-${provider}`,
      type: 'password',
      autocomplete: 'new-password',
      'data-1p-ignore': true,
      'data-lpignore': 'true',
      spellcheck: 'false',
      autocapitalize: 'off',
      ...props,
    });
  }

  /** @param {Props} props */
  function endpointField(props) {
    return h('input', {
      class: 'field',
      id: 'endpoint-other',
      form: 'settings-form',
      type: 'text',
      inputmode: 'url',
      pattern: 'https?://\\S+',
      title: 'starts with http:// or https://',
      spellcheck: 'false',
      autocapitalize: 'off',
      placeholder: 'http://localhost:8000/v1',
      ...props,
    });
  }

  // A provider's row is a label, a field and, below it, its state; the servers follow, then a server to add.
  function keyRows() {
    const rows = Object.entries(PROVIDERS).flatMap(([provider, { label }], i) => {
      const marker = h('span', { id: `state-${provider}` });
      const forgetIt = () => forget(() => mode.forgetKey(provider), label, `key-${provider}`);
      const remove = h('button', { class: 'btn remove', id: `remove-${provider}`, type: 'button', 'aria-label': `remove ${label} key`, onclick: forgetIt }, 'remove');
      const state = h('div', { class: 'k key-state', id: `key-state-${provider}`, hidden: true }, marker, remove);
      const key = keyField(provider, { autofocus: i === 0, 'aria-describedby': marker.id });
      return [h('label', { for: key.id }, `${label} API key`), key, state];
    });
    const servers = h('ul', { class: 'k server-list', id: 'server-list', 'aria-labelledby': 'servers-label', hidden: true });
    const endpoint = endpointField({ 'aria-label': 'add server: URL', 'aria-describedby': 'other-hint' });
    const key = keyField('other', { placeholder: 'API key, if any', 'aria-label': 'add server: API key' });
    return [...rows, h('span', { class: 'keys-label', id: 'servers-label', hidden: true }, 'Servers'), servers, h('label', { for: endpoint.id }, 'add server'), endpoint, key];
  }

  // Storage is outside input: anything but a count r3con app takes reads as the default.
  function readRounds() {
    const rounds = Number(readStored(ROUNDS_KEY));
    return Number.isInteger(rounds) && rounds >= 1 && rounds <= MAX_ROUNDS ? rounds : DEFAULT_ROUNDS;
  }

  const pickedRounds = () => Number(roundsPick.value);

  function buildSettings() {
    fill($('keys'), keyRows());
    fill(roundsPick, range(MAX_ROUNDS).map((k) => h('option', { value: k + 1 }, k + 1 === DEFAULT_ROUNDS ? `${k + 1} · default` : String(k + 1))));
    roundsPick.value = String(readRounds());
  }

  function clearSettings() {
    for (const input of $('settings').querySelectorAll('input')) input.value = '';
    $('endpoint-other', HTMLInputElement).required = false;
  }

  // Right-aligned under the button; CSS owns the width, so nothing needs measuring before it opens.
  function placeSettings() {
    const button = $('settings-btn').getBoundingClientRect();
    const popover = $('settings');
    popover.style.left = `max(var(--gutter), ${button.right}px - var(--settings-width))`;
    popover.style.top = `${button.bottom + POPOVER_GAP}px`;
    popover.style.maxHeight = `${window.innerHeight - button.bottom - POPOVER_GAP - POPOVER_MARGIN}px`;
  }

  // beforetoggle, not toggle, so the popover is placed and cleared before its first paint.
  /** @param {ToggleEvent} e */
  function onSettingsToggle(e) {
    clearSettings();
    if (e.newState !== 'open') return;
    placeSettings();
    showSaveResult('');
  }

  let savedTimer = 0;

  /** @param {string} text */
  function showSaveResult(text) {
    clearTimeout(savedTimer);
    $('settings-saved').textContent = text;
  }

  let saving = false;

  // A save asked for while one is in flight is dropped: it would send the same keys again.
  async function saveSettings() {
    if (saving) return;
    saving = true;
    try {
      await saveEnteredKeys();
    } finally {
      saving = false;
    }
  }

  // One at a time, so a failure stops the save with the fields not yet saved still holding what was typed.
  async function saveEnteredKeys() {
    const endpoint = $('endpoint-other', HTMLInputElement).value;
    const entered = Object.keys(PROVIDERS)
      .map((provider) => [provider, $(`key-${provider}`, HTMLInputElement).value.trim()])
      .filter(([, key]) => key);
    if (!entered.length && !endpoint) return showSaveResult('nothing to save');
    for (const [provider, key] of entered) {
      try {
        renderKeys(await mode.saveKey(provider, key));
      } catch (error) {
        return showSaveResult(`${PROVIDERS[provider].label} · ${explain(error, "couldn't save", "couldn't save the keys")}`);
      }
      $(`key-${provider}`, HTMLInputElement).value = '';
    }
    if (endpoint) {
      try {
        renderKeys(await mode.saveServer(endpoint, $('key-other', HTMLInputElement).value.trim()));
      } catch (error) {
        return showSaveResult(`${SERVER_LABEL} · ${explain(error, "couldn't save", "couldn't save the server")}`);
      }
    }
    clearSettings();
    showDone('saved');
  }

  /** @param {string} text */
  function showDone(text) {
    showSaveResult(text);
    savedTimer = setTimeout(() => showSaveResult(''), SAVED_MS);
  }

  // Its button goes with what it removes, so focus moves to the field that was typed in.
  /** @param {() => Promise<KeyStatus>} request @param {string} label @param {string} field */
  async function forget(request, label, field) {
    try {
      renderKeys(await request());
    } catch (error) {
      return showSaveResult(`${label} · ${explain(error, "couldn't remove", "couldn't remove a key")}`);
    }
    $(field).focus();
    showDone('removed');
  }

  // ---------- gestures ----------

  /**
   * @type {{ pointers: Map<number, Point>, mode: 'pinch' | 'drag' | 'pan' | null, moved: boolean, origin: Point,
   *   from: Point, node: PanelNode | null, pinch: Pinch | null }}
   */
  const gesture = { pointers: new Map(), mode: null, moved: false, origin: { x: 0, y: 0 }, from: { x: 0, y: 0 }, node: null, pinch: null };

  /** @param {number} x @param {number} y */
  function panelPoint(x, y) {
    const r = panel.getBoundingClientRect();
    return { x: x - r.left, y: y - r.top };
  }

  function startPinch() {
    const { view } = state.shown;
    trash.classList.remove('over');
    const [a, b] = [...gesture.pointers.values()];
    gesture.mode = 'pinch';
    gesture.pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y) || 1, mid: panelPoint((a.x + b.x) / 2, (a.y + b.y) / 2), view: { ...view } };
  }

  function updatePinch() {
    const { view } = state.shown;
    const [a, b] = [...gesture.pointers.values()];
    const p = /** @type {Pinch} */ (gesture.pinch);
    const k = clamp((p.view.k * Math.hypot(a.x - b.x, a.y - b.y)) / p.dist, ZOOM.min, ZOOM.max);
    const mid = panelPoint((a.x + b.x) / 2, (a.y + b.y) / 2);
    const wx = (p.mid.x - p.view.x) / p.view.k;
    const wy = (p.mid.y - p.view.y) / p.view.k;
    view.k = k;
    view.x = mid.x - wx * k;
    view.y = mid.y - wy * k;
    state.shown.userMoved = true;
    applyView();
  }

  // A document node of the shown tab, released over the can while it shows, has its file removed.
  /** @param {PointerEvent} e */
  function overTrash(e) {
    const { node } = gesture;
    const droppable = gesture.mode === 'drag' && gesture.moved && !trash.hidden && node?.kind === 'doc' && !node.fold;
    if (!droppable) return false;
    const r = trash.getBoundingClientRect();
    return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
  }

  /** @param {PointerEvent} e */
  function onPointerDown(e) {
    const { view } = state.shown;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    if (targetOf(e).closest('.inspector, .panel-tools, .remove-all')) return;
    gesture.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (gesture.pointers.size === 2) return startPinch();
    if (gesture.pointers.size > 2) return;
    const id = hoverFrom(e);
    gesture.node = id ? drawnNode(state.shown, id) : null;
    gesture.mode = gesture.node ? 'drag' : 'pan';
    gesture.moved = false;
    gesture.origin = { x: e.clientX, y: e.clientY };
    gesture.from = gesture.node ? drawnAt(gesture.node) : { x: view.x, y: view.y };
  }

  /** @param {PointerEvent} e */
  function onPointerMove(e) {
    const { view } = state.shown;
    const p = gesture.pointers.get(e.pointerId);
    if (!p) return;
    // A mouse released outside the panel before the gesture was captured sent its pointerup elsewhere.
    if (e.pointerType === 'mouse' && !e.buttons) return onPointerEnd(e);
    p.x = e.clientX;
    p.y = e.clientY;
    if (gesture.mode === 'pinch') return updatePinch();
    if (!gesture.mode) return;
    const dx = e.clientX - gesture.origin.x;
    const dy = e.clientY - gesture.origin.y;
    if (!gesture.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
    if (!gesture.moved) panel.setPointerCapture(e.pointerId);
    gesture.moved = true;
    if (gesture.mode === 'drag') {
      const node = /** @type {PanelNode} */ (gesture.node);
      state.shown.placed.add(node.id);
      node.x = gesture.from.x + dx / view.k;
      node.y = gesture.from.y + dy / view.k;
      placeNode(node);
      redrawEdges(state.shown);
      if (state.selected === node.id) positionInspector();
      trash.classList.toggle('over', overTrash(e));
    } else {
      panel.classList.add('panning');
      view.x = gesture.from.x + dx;
      view.y = gesture.from.y + dy;
      state.shown.userMoved = true;
      applyView();
    }
  }

  function endGesture() {
    gesture.pointers.clear();
    gesture.mode = null;
    gesture.node = null;
    trash.classList.remove('over');
    panel.classList.remove('panning');
  }

  /** @param {PointerEvent} e */
  function onPointerEnd(e) {
    if (!gesture.pointers.delete(e.pointerId)) return;
    const trashed = e.type === 'pointerup' && overTrash(e) ? gesture.node?.file : null;
    // A pinch goes on while two fingers are left, measured afresh: they may be another pair.
    if (gesture.mode === 'pinch') {
      if (gesture.pointers.size < 2) endGesture();
      else startPinch();
      return;
    }
    if (e.type === 'pointerup' && !gesture.moved && gesture.mode === 'pan') closeInspector();
    endGesture();
    if (!trashed) return;
    removeFile(trashed, { refocus: false });
    // The dragged node held the focus, and went with its file.
    if (document.activeElement === document.body) panel.focus({ preventScroll: true });
  }

  /** @param {WheelEvent} e */
  function onWheel(e) {
    const { view } = state.shown;
    if (targetOf(e).closest('.inspector')) return;
    e.preventDefault();
    const scale = e.deltaMode === 1 ? WHEEL_LINE_PX : 1;
    if (e.ctrlKey || e.metaKey) {
      const p = panelPoint(e.clientX, e.clientY);
      zoomAt(p.x, p.y, view.k * Math.exp(-e.deltaY * scale * WHEEL_ZOOM_RATE));
    } else {
      view.x -= e.deltaX * scale;
      view.y -= e.deltaY * scale;
      applyView();
    }
    state.shown.userMoved = true;
  }

  const inspectorGrab = { x: 0, y: 0, left: 0, top: 0 };

  /** @param {PointerEvent} e */
  function onInspectorGrab(e) {
    if (!targetOf(e).closest('.insp-head') || targetOf(e).closest('button')) return;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    Object.assign(inspectorGrab, { x: e.clientX, y: e.clientY, left: inspector.offsetLeft, top: inspector.offsetTop });
    inspector.setPointerCapture(e.pointerId);
  }

  /** @param {PointerEvent} e */
  function onInspectorMove(e) {
    if (!inspector.hasPointerCapture(e.pointerId)) return;
    const dx = e.clientX - inspectorGrab.x;
    const dy = e.clientY - inspectorGrab.y;
    if (!inspector.classList.contains('moving') && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
    inspector.classList.add('moving');
    state.inspectorDroppedAt = { left: inspectorGrab.left + dx, top: inspectorGrab.top + dy };
    placeInspector(state.inspectorDroppedAt);
  }

  /** @param {Event} e */
  function hoverFrom(e) {
    /** @type {HTMLElement | null} */
    const el = targetOf(e).closest('.node');
    return el ? /** @type {string} */ (el.dataset.id) : null;
  }

  // ---------- side column: its width, from the line between it and the panel ----------

  /** @type {number | null} */
  let sideWidth = Number(readStored(SIDE_WIDTH_KEY)) || null;
  let sideMax = SIDE_MIN_PX;
  const sideGrab = { x: 0, width: 0, moved: false, movedBefore: false };

  // The width chosen, within what the frame allows now; with none chosen, the stylesheet's.
  function showSideWidth() {
    const width = sideWidth && clamp(sideWidth, SIDE_MIN_PX, sideMax);
    if (width) frame.style.setProperty('--side-width', `${width}px`);
    else frame.style.removeProperty('--side-width');
    resizer.ariaValueMax = String(sideMax);
    resizer.ariaValueNow = String(width || side.offsetWidth);
  }

  // The widest the frame allows now, never under the narrowest, and the width within it.
  function fitSideToFrame() {
    sideMax = Math.max(SIDE_MIN_PX, Math.round(frame.clientWidth * SIDE_MAX_SHARE));
    showSideWidth();
  }

  /** @param {number | null} width */
  function setSideWidth(width) {
    sideWidth = width === null ? null : clamp(Math.round(width), SIDE_MIN_PX, sideMax);
    showSideWidth();
  }

  function keepSideWidth() {
    if (sideWidth) writeStored(SIDE_WIDTH_KEY, String(sideWidth));
    else removeStored(SIDE_WIDTH_KEY);
  }

  /** @param {PointerEvent} e */
  function onResizerGrab(e) {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    Object.assign(sideGrab, { x: e.clientX, width: side.offsetWidth, moved: false, movedBefore: sideGrab.moved });
    resizer.setPointerCapture(e.pointerId);
    frame.classList.add('resizing');
  }

  // A drag only sets the width, reading no layout, and keeps it once, on release.
  /** @param {PointerEvent} e */
  function onResizerMove(e) {
    if (!resizer.hasPointerCapture(e.pointerId)) return;
    const dx = sideGrab.x - e.clientX;
    if (!sideGrab.moved && Math.abs(dx) < DRAG_THRESHOLD) return;
    sideGrab.moved = true;
    setSideWidth(sideGrab.width + dx);
  }

  function onResizerDrop() {
    frame.classList.remove('resizing');
    if (sideGrab.moved) keepSideWidth();
  }

  // The line moves the way the arrow points; with a modifier held, the key is the browser's.
  /** @param {KeyboardEvent} e */
  function onResizerKey(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const width = side.offsetWidth;
    const to = { ArrowLeft: width + SIDE_STEP_PX, ArrowRight: width - SIDE_STEP_PX, Home: SIDE_MIN_PX, End: sideMax }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    setSideWidth(to);
    keepSideWidth();
  }

  // Pointer capture sends both clicks here even when they dragged, so only two that didn't reset it.
  function resetSideWidth() {
    if (sideGrab.moved || sideGrab.movedBefore) return;
    setSideWidth(null);
    keepSideWidth();
  }

  // ---------- wiring ----------

  function wire() {
    panel.addEventListener('pointerdown', onPointerDown);
    panel.addEventListener('pointermove', onPointerMove);
    panel.addEventListener('pointerup', onPointerEnd);
    panel.addEventListener('pointercancel', onPointerEnd);
    panel.addEventListener('wheel', onWheel, { passive: false });
    inspector.addEventListener('pointerdown', onInspectorGrab);
    inspector.addEventListener('pointermove', onInspectorMove);
    inspector.addEventListener('lostpointercapture', () => inspector.classList.remove('moving'));
    resizer.addEventListener('pointerdown', onResizerGrab);
    resizer.addEventListener('pointermove', onResizerMove);
    resizer.addEventListener('lostpointercapture', onResizerDrop);
    resizer.addEventListener('keydown', onResizerKey);
    resizer.addEventListener('dblclick', resetSideWidth);

    world.addEventListener('pointerover', (e) => {
      state.hovered = hoverFrom(e);
      refreshHighlight();
    });
    world.addEventListener('pointerout', (e) => {
      if (e.relatedTarget && targetOf(e).closest('.node')?.contains(/** @type {Node} */ (e.relatedTarget))) return;
      state.hovered = null;
      refreshHighlight();
    });
    world.addEventListener('focusin', (e) => {
      state.hovered = hoverFrom(e);
      refreshHighlight();
    });
    world.addEventListener('focusout', () => {
      state.hovered = null;
      refreshHighlight();
    });
    // An animation's class comes off once it has played, so the node or line can play it again.
    world.addEventListener('animationend', (e) => endAnimation(targetOf(e)));
    // On click, not pointerup: a tap's click would otherwise land on the inspector that just opened under it.
    panel.addEventListener('click', (e) => {
      const id = hoverFrom(e);
      // element.click() and assistive tech send detail 0: no pointer behind them, so no drag to skip.
      const endsDrag = gesture.moved && e.detail > 0;
      if (id && !endsDrag) openInspector(id);
    });
    world.addEventListener('keydown', (e) => {
      const id = hoverFrom(e);
      if (id && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        openInspector(id, { focus: true });
      }
    });

    document.addEventListener('keydown', (e) => {
      // The platform closes an open settings popover on this same Escape.
      if (e.key !== 'Escape' || $('settings').matches(':popover-open')) return;
      if (!inspector.hidden) closeInspector(true);
      else if (!$('file-view').hidden) closeFile();
    });

    $('arrange-btn').addEventListener('click', arrange);
    $('settings').addEventListener('beforetoggle', onSettingsToggle);
    window.addEventListener('resize', () => $('settings').matches(':popover-open') && placeSettings());
    $('settings').addEventListener('input', () => showSaveResult(''));
    $('key-other').addEventListener('input', () => ($('endpoint-other', HTMLInputElement).required = Boolean($('key-other', HTMLInputElement).value.trim())));
    // The key fields sit outside the form, so Enter there doesn't submit it on its own.
    $('settings').addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' || e.isComposing || !targetOf(e).matches('.key')) return;
      e.preventDefault();
      $('settings-form', HTMLFormElement).requestSubmit();
    });
    $('settings-form').addEventListener('submit', (e) => {
      e.preventDefault();
      saveSettings();
    });
    $('fit-btn').addEventListener('click', refit);
    $('replay-btn').addEventListener('click', () => state.shown.run && playRun(runTab(state.shown)));
    $('run-tabs').addEventListener('keydown', onTabKey);
    $('new-tab-btn').addEventListener('click', () => {
      const from = state.shown;
      const tab = copyTab(from);
      switchTab(tab, docPositions(from));
      if (tab.files.length) {
        showTab('chat');
        questionEl.focus();
      } else {
        showTab('files');
        tab.button.focus();
      }
    });
    for (const button of removeAllButtons) button.addEventListener('click', removeAllOrUndo);
    $('chat-pane').addEventListener('click', onChatClick);
    $('theme-btn').addEventListener('click', toggleTheme);

    $('tab-chat').addEventListener('click', () => showTab('chat'));
    $('tab-files').addEventListener('click', () => {
      showTab('files');
      closeFile();
    });

    $('ask-form').addEventListener('submit', (e) => {
      e.preventDefault();
      submitQuestion(primaryAction());
    });
    $('new-run-btn').addEventListener('click', () => submitQuestion('run'));
    $('stop-btn').addEventListener('click', () => {
      mode.stop(state.shown);
      updateComposer();
    });
    questionEl.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) submitQuestion('run');
      else $('ask-form', HTMLFormElement).requestSubmit();
    });
    modelPick.addEventListener('change', onPick);
    modelOther.addEventListener('change', onPick);
    // In the form, Enter would also send the question.
    modelOther.addEventListener('keydown', (e) => {
      if (e.isComposing) return;
      if (e.key === 'Enter') {
        e.preventDefault();
        saveTypedModel();
      } else if (e.key === 'Escape') {
        e.stopPropagation();
        cancelTypedModel();
      }
    });
    modelRemove.addEventListener('click', removeSavedModel);
    roundsPick.addEventListener('change', () => writeStored(ROUNDS_KEY, roundsPick.value));
    questionEl.addEventListener('input', readyStatus);

    for (const id of ['file-input', 'folder-input']) {
      const input = $(id, HTMLInputElement);
      input.addEventListener('change', () => {
        // Copied now: a file input's list empties once its value is reset.
        const files = /** @type {FileList} */ (input.files);
        if (files.length) addFiles(fromFolders([...files]));
        input.value = '';
      });
    }

    /** @param {DragEvent} e @returns {e is DragEvent & { dataTransfer: DataTransfer }} */
    const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');
    // The panel and the side column take files alike. A node dragged in the panel is pointer events, never a drag of files.
    // The overlay is on while the drag is over any element in the frame, counted, as WebKit sends no relatedTarget.
    let dragDepth = 0;
    frame.addEventListener('dragenter', (e) => {
      if (!hasFiles(e)) return;
      dragDepth += 1;
      frame.classList.add('dropping');
    });
    frame.addEventListener('dragleave', (e) => {
      if (!hasFiles(e)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) frame.classList.remove('dropping');
    });
    frame.addEventListener('dragover', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });
    frame.addEventListener('drop', async (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      const tab = state.shown;
      showTab('files');
      // A tab that has run keeps its run's files, so what is dropped on it is refused unread.
      if (tab.run) return tellFiles(tab, { filesError: notAdded(pickFiles([...e.dataTransfer.files]).map((f) => f.name), LOCKED_DROP) });
      addFiles(await filesFromDrop(e.dataTransfer), tab);
    });
    // However a drag ends, dropped anywhere or called off, the overlay goes, whatever dragleave was missed.
    const dragEnded = () => {
      dragDepth = 0;
      frame.classList.remove('dropping');
    };
    window.addEventListener('dragover', (e) => hasFiles(e) && e.preventDefault());
    window.addEventListener('drop', (e) => {
      dragEnded();
      if (hasFiles(e)) e.preventDefault();
    });
    window.addEventListener('dragend', dragEnded);

    new ResizeObserver(() => {
      fitShown();
      positionInspector();
    }).observe(panel);
    new ResizeObserver(fitSideToFrame).observe(frame);
    new ResizeObserver(positionInspector).observe(inspector);
  }

  // ---------- modes ----------
  // What differs between the page served by r3con app and the prototype, decided once at load.

  /** @type {Mode} */
  const prototypeMode = {
    note: PROTOTYPE_MODE_NOTE,
    keysNote: PROTOTYPE_KEYS_NOTE,
    nextRun: (question) => (replaysRecording(question) ? `recorded run · ${SAMPLE.model}` : `simulated run · ${pickedModel() || 'type a model'}`),
    needsModel: (question, asks) => asks || !replaysRecording(question),
    refuseFiles: () => null,
    extract: async () => ({ why: NEEDS_APP }),
    unzip: async (file) => oneDocument(file, { why: NEEDS_APP }),
    newRun: (question, files) => (replaysRecording(question, files) ? sampleRun() : simulatedRun(files, question, pickedModel(), pickedRounds())),
    play: playTimeline,
    // A replay is going once it was started with motion, and not cancelled, until its answer lands.
    runGoing: (tab) => tab.timers.length > 0 && !landed(tab, RUN_ANSWER),
    // It stops where it is, as a live run does once its call in flight is back.
    stop(tab) {
      cancelTimers(tab);
      const narr = /** @type {Narrator} */ (tab.narr);
      failRun(tab, narr, { outcome: inStage('stopped', narr.stage), error: '' });
    },
    ask: (tab, question) => prototypeAsk(tab, question, pickedModel()),
    async saveKey(provider, key) {
      keyed.set(provider, { source: 'here', hint: hintOf(key) });
      return prototypeStatus();
    },
    async saveServer(endpoint, key) {
      const refusal = serverUrlRefusal(normalEndpoint(endpoint));
      if (refusal) throw new ServerError(`other.endpoint: ${refusal}`);
      prototypeServers.set(normalEndpoint(endpoint), key ? hintOf(key) : null);
      return prototypeStatus();
    },
    async forgetKey(provider) {
      keyed.delete(provider);
      return prototypeStatus();
    },
    async forgetServer(endpoint) {
      prototypeServers.delete(endpoint);
      return prototypeStatus();
    },
    readKeys: async () => prototypeStatus(),
  };

  let extracting = 0;
  /** @type {(() => void)[]} */
  const extractionsWaiting = [];

  // Runs `extraction` once fewer than EXTRACTIONS_AT_ONCE are going, in the order they were asked for.
  /** @template T @param {() => Promise<T>} extraction @returns {Promise<T>} */
  async function inTurn(extraction) {
    if (extracting < EXTRACTIONS_AT_ONCE) extracting += 1;
    else await new Promise((resolve) => extractionsWaiting.push(() => resolve(undefined)));
    try {
      return await extraction();
    } finally {
      // A slot passes straight to the next in line, so a later call can't take it first.
      const next = extractionsWaiting.shift();
      if (next) next();
      else extracting -= 1;
    }
  }

  // A provider whose variable is set where r3con app runs counts as having a key.
  /** @type {Mode} */
  const liveMode = {
    note: '',
    keysNote: '',
    nextRun: () => pickedModel() || 'type a model',
    needsModel: () => true,
    refuseFiles: unreadableFiles,
    async extract(file, noText) {
      try {
        const { text } = await inTurn(async () => (await api(`/api/documents/text?name=${encodeURIComponent(file.name)}`, { method: 'POST', body: file })).json());
        return text.trim() ? { text } : { why: noText };
      } catch (error) {
        return { why: `Couldn't get the text: ${explain(error, /** @type {Error} */ (error).message, "couldn't get a file's text")}.` };
      }
    },
    // A zipped document with no text is refused as a dropped one is, and so is a zip with none.
    async unzip(file) {
      try {
        /** @type {{ files: { name: string, size: number, text: string }[], skipped: number }} */
        const { files, skipped } = await inTurn(async () => (await api(`/api/documents/zip?name=${encodeURIComponent(file.name)}`, { method: 'POST', body: file })).json());
        if (!files.length) return oneDocument(file, { why: noDocuments(skipped) });
        const noText = (/** @type {{ name: string }} */ f) => (hasSuffix(f, PDF_SUFFIX) ? NO_TEXT_LAYER : NO_TEXT);
        return { docs: files.map((f) => (f.text.trim() ? f : { name: f.name, size: f.size, why: noText(f) })), skipped };
      } catch (error) {
        return oneDocument(file, { why: `Couldn't unzip it: ${explain(error, /** @type {Error} */ (error).message, "couldn't unzip a file")}.` });
      }
    },
    newRun: (question, files) => liveRun(question, pickedModel(), pickedEndpoint(), files, pickedRounds()),
    play: streamRun,
    runGoing: (tab) => streaming('run', tab),
    stop: (tab) => toStop(tab).forEach(stopJob),
    ask: (tab, question) => askOver(tab, question, pickedModel(), pickedEndpoint()),
    saveKey: async (provider, key) => (await api('/api/keys', { method: 'POST', body: { provider, key } })).json(),
    saveServer: async (endpoint, key) => (await api('/api/keys', { method: 'POST', body: { provider: 'other', endpoint, key: key || undefined } })).json(),
    forgetKey: async (provider) => (await api('/api/keys', { method: 'DELETE', body: { provider } })).json(),
    forgetServer: async (endpoint) => (await api('/api/keys', { method: 'DELETE', body: { provider: 'other', endpoint } })).json(),
    readKeys: async () => (await api('/api/keys')).json(),
  };

  // r3con refuses a blank document, so a run with one is not sent.
  /** @param {LoadedFile[]} files */
  function unreadableFiles(files) {
    const names = files.filter((f) => !f.text.trim()).map((f) => f.name);
    return names.length ? `no readable text in ${someNames(names)}; nothing was sent` : null;
  }

  const mode = live ? liveMode : prototypeMode;

  function init() {
    restoreTheme();
    resizer.ariaValueMin = String(SIDE_MIN_PX);
    fitSideToFrame();
    removeStored(RETIRED_SETTINGS_KEY);
    buildSettings();
    questionEl.value = SAMPLE.question;
    openTab(state.shown);
    showTabContents(state.shown);
    syncDocNodes(state.shown);
    updateComposer();
    renderPicker();
    readyStatus();
    refreshKeys().catch((error) => errorStatus(explain(error, "couldn't read which providers have keys")));
    $('keys-note').textContent = mode.keysNote;
    $('keys-note').hidden = !mode.keysNote;
    $('files-locked').textContent = FILES_LOCKED;
    $('mode-note').textContent = mode.note;
    $('mode-note').hidden = !mode.note;
    wire();
  }

  init();
})();
