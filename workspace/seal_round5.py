"""Generate the new sealed 36-seed final pool without displaying attributes/results."""
from pathlib import Path
from round4_common import ROOT, save, digest, file_hash
from round4_seeds import generate,FAMILIES

OUT=ROOT/'agent_notes/experiments/round5'


def main():
    if (OUT/'sealed-final-manifest.json').exists():raise RuntimeError('Final pool already frozen')
    entries=[]
    for f,family in enumerate(FAMILIES):
        for strength,scale in enumerate((.55,1.,1.5)):
            for rep in range(2):
                index=len(entries)
                spec={'case':f'round5-final-{index:03}','pool':'round5-final','family':family,'scale':scale,
                    'persona_mix':(f+strength+rep)%3,'rep':rep,'mode':'independent' if rep==0 else 'w_r','n':50,'ood':False,'generator_version':5}
                s=generate(spec)
                save(OUT/'sealed-final-seeds'/(spec['case']+'.json'),s)
                entries.append({'case':spec['case'],'seed_sha256':digest(s),'base_graph_group':spec['case'],
                    'stratum':[family,strength],'glm_subset':strength==1})
    save(OUT/'sealed-final-manifest.json',{'generator_sha256':file_hash(__file__),'entries':entries,
        'n':36,'glm_subset_n':12,'purpose':'sealed independent synthetic validation after candidate freeze',
        'attribute_results_viewed':False,'generation_note':'Independent RNG streams include round5 case/spec identity; response association is synthetic and explicitly specified by generator.'})
    print('Sealed 36 new seeds and preselected 12 GLM subset; no attributes or scores displayed.')


if __name__=='__main__':main()
