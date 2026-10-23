"""Narrow literal conditional-threat rule; never a keyword-only classifier."""
import re

def threat_rule(text):
 # A narrowly scoped fallback for invitation/condition + first-person violent response.
 # No keyword alone can fire; normal, quoted, negated and fictional contexts block it.
 s=' '.join(text.lower().split())
 exclusions=r'\b(news|reported|reporting|novel|fiction|fictional|movie|film|game|photograph\w*|camera|portrait\w*|landscape\w*|sunset|basketball|hoops|pool|billiards|skeet|clay pigeons|paper targets?|shooting range|condemns?|opposes?|do not|don.t|never|not going to)\b|新闻|通报|小说|虚构|电影|游戏|摄影|相机|靶场|纸靶|反对|不要|不会'
 if re.search(exclusions,s):return None
 cue=re.search(r'\b(?:please|if you|when you|you|whoever|anyone who)\s+(?:loot|steal|rob|trespass)\b',s)
 harm=re.search(r'\b(?:i|we)\s+(?:will\s+|shall\s+|am going to\s+|are going to\s+|love to\s+)?(?:shoot|kill|stab|beat)\b',s)
 if cue and harm and cue.start()<harm.start() and harm.end()-cue.start()<=180:
  return dict(rule_id='conditional_first_person_violence_v1',evidence=text,reason='条件或邀请行为与第一人称伤害响应形成同一段威胁，原文未给出新闻、反对或虚构等解释。')
 return None
