"""Materialize document-grounded annotations before examining retrieval ranks.

Pages refer to one-based PDF pages. These target necessary exception clauses,
not exhaustive relevance judgments. Keep the full pool when reporting results.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# category, article, page, necessary verbatim clause, query variants
specs = [
    ("negation_constraints", "Article 6(1), final subparagraph", 36,
     "Point (f) of the first subparagraph shall not apply to processing carried out by public authorities in the performance of their tasks.", [
         "Can a public authority rely on legitimate interests when processing personal data in the performance of its tasks?",
         "A municipal authority wants to use legitimate interests rather than a statutory basis for its official duties. Is that basis available?",
         "Is legitimate interest a lawful basis for government bodies carrying out their public tasks?",
     ]),
    ("exceptions_corner_cases", "Article 30(5)", 51,
     "the processing is not occasional", [
         "Does an organisation with fewer than 250 employees always avoid keeping records of processing activities? What if processing is not occasional?",
         "Our company has 100 staff and processes personal information every day. Does the small-company record-keeping exemption apply?",
         "Are small employers automatically exempt from processing records when their processing is routine rather than occasional?",
     ]),
    ("exceptions_corner_cases", "Article 34(3)(a)", 53,
     "render the personal data unintelligible to any person who is not authorised to access it, such as encryption", [
         "Must affected individuals always be told about a high-risk breach if the stolen data was made unintelligible to unauthorised people by encryption?",
         "When is individual breach communication not required because the compromised data cannot be read by unauthorised persons?",
         "If a laptop containing encrypted customer data is stolen, what protection-measures exception may remove the need to tell each customer?",
     ]),
    ("exceptions_corner_cases", "Article 34(3)(c)", 53,
     "it would involve disproportionate effort. In such a case, there shall instead be a public communication or similar measure whereby the data subjects are informed in an equally effective manner.", [
         "If informing every affected individual about a data breach would involve disproportionate effort, may the controller simply do nothing?",
         "What substitute communication must a controller make when telling people individually about a breach is disproportionately difficult?",
         "Is a public announcement permitted instead of contacting each victim of a breach, and on what condition?",
     ]),
    ("exceptions_corner_cases", "Article 12(3)", 40,
     "That period may be extended by two further months where necessary, taking into account the complexity and number of the requests.", [
         "Must a controller finish every subject rights request within one month even if there are many complex requests?",
         "A business needs extra time to respond to a large number of complicated access requests. How long can it extend the deadline?",
         "When may the one-month deadline for responding to an individual's rights request be extended, and by how much?",
     ]),
    ("keyword_trap_distractor", "Article 12(5)", 40,
     "The controller shall bear the burden of demonstrating the manifestly unfounded or excessive character of the request.", [
         "When a controller refuses a repetitive or excessive data access request, who has to prove that the request is excessive?",
         "Who must demonstrate that a subject access request is manifestly unfounded when a company charges a fee?",
         "Can a company require a requester to prove their repeated access request is reasonable before charging an administrative fee?",
     ]),
    ("keyword_trap_distractor", "Article 57(4)", 69,
     "The supervisory authority shall bear the burden of demonstrating the manifestly unfounded or excessive character of the request.", [
         "Who bears the burden of proof when a supervisory authority charges a fee for a repetitive request?",
         "A privacy regulator refuses to act on an excessive request. Does the regulator or the requester need to prove excessiveness?",
         "When requests to the supervisory authority are manifestly unfounded, which party has to substantiate that assessment?",
     ]),
    ("negation_constraints", "Article 21(3)", 45,
     "Where the data subject objects to processing for direct marketing purposes, the personal data shall no longer be processed for such purposes.", [
         "Can a controller keep using personal data for direct marketing after an objection by citing compelling legitimate grounds?",
         "After I object to direct marketing, can a company override my objection because its commercial interests are stronger?",
         "Does the compelling legitimate grounds exception let an organisation continue advertising to someone who has objected?",
     ]),
    ("negation_constraints", "Article 7(3)", 37,
     "The withdrawal of consent shall not affect the lawfulness of processing based on consent before its withdrawal.", [
         "Does withdrawing consent make all processing that previously relied on that consent unlawful retroactively?",
         "If a person revokes permission today, was the company's earlier consent-based data use illegal?",
         "What happens to the lawfulness of past data processing when consent is later withdrawn?",
     ]),
    ("exceptions_corner_cases", "Article 8(1), second subparagraph", 37,
     "provided that such lower age is not below 13 years", [
         "Is sixteen always the minimum age at which a child may consent to an information society service, or may a Member State lower it?",
         "What is the lowest age a Member State can choose for a child's own consent to an online service?",
         "Can national law allow a thirteen-year-old to consent to an information society service without parental authorisation?",
     ]),
    ("negation_constraints", "Article 20(3)", 45,
     "That right shall not apply to processing necessary for the performance of a task carried out in the public interest or in the exercise of official authority vested in the controller.", [
         "Does the data portability right apply to processing necessary for a task in the public interest or official authority?",
         "Can I demand portable data from a public body processing it to perform its statutory public task?",
         "Is processing for an official public-interest task covered by the right to receive and transfer my personal data?",
     ]),
    ("document_specific", "Article 37(1)(a)", 55,
     "except for courts acting in their judicial capacity", [
         "Must courts always appoint a data protection officer when acting in their judicial capacity?",
         "Which public authority is excepted from the mandatory data protection officer rule when it performs judicial functions?",
         "Does the rule requiring public bodies to designate a DPO also cover a court's judicial work?",
     ]),
    ("negation_constraints", "Article 55(3)", 67,
     "Supervisory authorities shall not be competent to supervise processing operations of courts acting in their judicial capacity.", [
         "Can a supervisory authority supervise the personal-data processing of a court acting in its judicial capacity?",
         "Who is outside a data protection supervisory authority's competence when acting in a judicial role?",
         "Does a national privacy regulator have jurisdiction over a court's processing during judicial proceedings?",
     ]),
    ("exceptions_corner_cases", "Article 33(1)", 52,
     "unless the personal data breach is unlikely to result in a risk to the rights and freedoms of natural persons", [
         "Must every personal data breach be notified to the supervisory authority, even one unlikely to risk individuals' rights and freedoms?",
         "What risk-based exception allows a controller not to notify the regulator of a personal data breach?",
         "If a breach is unlikely to harm people's rights, is notification to the authority still mandatory within 72 hours?",
     ]),
    ("keyword_trap_distractor", "Article 33(2)", 52,
     "The processor shall notify the controller without undue delay after becoming aware of a personal data breach.", [
         "Does a processor have the same 72-hour deadline as a controller for telling the controller about a personal data breach?",
         "When a processor discovers a breach, whom must it inform and what deadline applies to that processor?",
         "May a data processor wait seventy-two hours before reporting a breach to its controller?",
     ]),
    ("document_specific", "Article 5(1)(b)", 35,
     "not be considered to be incompatible with the initial purposes", [
         "Is later use for scientific research automatically incompatible with the original purpose of collecting the personal data?",
         "Does using previously collected data for public-interest archiving necessarily breach purpose limitation?",
         "How does the purpose limitation rule treat further processing for historical research or statistics?",
     ]),
    ("exceptions_corner_cases", "Article 10", 39,
     "Any comprehensive register of criminal convictions shall be kept only under the control of official authority.", [
         "Who is permitted to control a comprehensive register of criminal convictions?",
         "Can a private company keep a comprehensive criminal-convictions register under its own exclusive control?",
         "Is consent sufficient for a company to independently control a comprehensive register of criminal convictions?",
     ]),
    ("negation_constraints", "Article 41(6)", 58,
     "This Article shall not apply to processing carried out by public authorities and bodies.", [
         "Does Article 41 monitoring of approved codes of conduct apply to processing by public authorities and bodies?",
         "Are public bodies' processing activities covered by the accredited code-of-conduct monitoring-body provisions?",
         "Can the code-of-conduct monitoring mechanism under Article 41 be applied to a public authority's processing?",
     ]),
]
# Equivalent evidence is labelled by reading the document, not by changing
# scores. Include recitals and repeated notices that state the same clause.
equivalents = {
    1: [(9, "that legal basis should not apply to the processing by public authorities in the performance of their tasks")],
    9: [(41, "without affecting the lawfulness of processing based on consent before its withdrawal"),
        (42, "without affecting the lawfulness of processing based on consent before its withdrawal")],
    11: [(13, "By its very nature, that right should not be exercised against controllers processing personal data in the exercise of their public duties."),
         (13, "It should therefore not apply where the processing of the personal data is necessary for compliance with a legal obligation to which the controller is subject or for the performance of a task carried out in the public interest or in the exercise of an official authority vested in the controller.")],
    12: [(18, "except for courts or independent judicial authorities when acting in their judicial capacity")],
    13: [(4, "The competence of the supervisory authorities should not cover the processing of personal data when courts are acting in their judicial capacity")],
    14: [(17, "unless the controller is able to demonstrate, in accordance with the accountability principle, that the personal data breach is unlikely to result in a risk to the rights and freedoms of natural persons")],
    16: [(9, "Further processing for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes should be considered to be compatible lawful processing operations.")],
}
queries = []
for topic, (category, article, page, evidence, variants) in enumerate(specs, 1):
    for variant, query in enumerate(variants, 1):
        modes = list(dict.fromkeys([category, "document_specific"] +
                    (["keyword_trap_distractor", "negation_constraints"] if topic == 16 else [])))
        queries.append({"id": f"q{topic:02d}-{variant}", "category": category,
                        "failure_modes": modes, "article": article,
                        "page": page, "evidence": evidence, "query": query,
                        "equivalent_evidence": [{"page": p, "evidence": e}
                                                for p, e in equivalents.get(topic, [])]})
(ROOT / "data" / "queries.json").write_text(json.dumps(queries, indent=2) + "\n")
print(f"Wrote {len(queries)} queries across {len(specs)} clause targets.")
