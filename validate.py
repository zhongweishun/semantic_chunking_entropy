"""Verify archive hashes, paired IDs, tree invariants and reference analysis outputs."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
from analysis.common import ROOT, read_json, write_json


def validate(check_results=False):
    provenance=read_json(ROOT/'data/provenance.json')
    for name,record in provenance['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==record['sha256'],name
    expected={r['story_id']:r for r in read_json(ROOT/'data/tree_manifest.json')}
    seen=set();total_tokens=0;multitoken=0;mismatches=[]
    with tarfile.open(ROOT/'data/trees_cut30.tar.gz') as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith('_result.json'):continue
            story_id=Path(member.name).name.split('_')[0]
            assert story_id not in seen,story_id
            seen.add(story_id)
            raw=tf.extractfile(member).read();tree=json.loads(raw);record=expected[story_id]
            assert member.name==record['member']
            assert hashlib.sha256(raw).hexdigest()==record['sha256'],story_id
            leaves=[]
            def walk(node):
                if isinstance(node,int):
                    assert node>=0
                    leaves.append(node)
                else:
                    assert isinstance(node,list) and 1<=len(node)<=4
                    for child in node:walk(child)
            walk(tree['partition'])
            n=sum(leaves)
            assert n==record['partition_tokens'] and len(tree['tokens'])==record['saved_tokens']
            assert len(leaves)==record['leaves']
            total_tokens+=n;multitoken+=sum(v>1 for v in leaves)
            if n!=len(tree['tokens']):mismatches.append({'story_id':story_id,'partition_tokens':n,'saved_tokens':len(tree['tokens'])})
    assert len(seen)==925 and seen==set(expected)
    paired=read_json(ROOT/'data/llm_entropy_reddit925.json.gz')
    all_scores=read_json(ROOT/'data/llm_entropy_reddit1000.json.gz')
    assert set(paired)==seen and len(all_scores)==1000
    assert all(paired[story_id]==all_scores[story_id] for story_id in seen)
    for story_id,record in all_scores.items():
        ti=np.asarray(record['TI_cumulative_token'])
        assert ti.ndim==1 and len(ti)>1 and np.all(np.isfinite(ti)),story_id
    report={'tree_count':len(seen),'paired_score_count':len(paired),'figure1_score_count':len(all_scores),
            'partition_tokens':total_tokens,'multitoken_leaves':multitoken,'coverage_mismatches':mismatches,
            'all_source_hashes_verified':True,'all_member_hashes_verified':True,'paired_scores_identical_to_full_cache':True}
    if check_results:
        a=read_json(ROOT/'results/figure1.json');b=read_json(ROOT/'results/figure2b.json');c=read_json(ROOT/'results/figure3.json')
        assert abs(a['slope']-2.546096409)<1e-7
        assert abs(a['r_squared']-.9717736929444166)<1e-10
        assert abs(b['mean_KL_L2_9']-.0501)<5e-5
        assert abs(c['LLM_mean']-2.820087466460249)<1e-10
        assert abs(c['LLM_std']-.405062677554728)<1e-10
        assert abs(c['tree_mean']-2.692)<.0005
        assert abs(c['tree_std']-.192)<.0005
        assert abs(c['correlation']-.270)<.0005
        for stem in ['figure1_entropy','figure2b_chunk_sizes','figure3_typicality','figure5abcd_scaling']:
            assert (ROOT/'figures'/f'{stem}.png').stat().st_size>10000
            assert (ROOT/'figures'/f'{stem}.pdf').stat().st_size>5000
        report['numerical_reference_checks_passed']=True
    write_json(ROOT/'results/validation.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',action='store_true')
    validate(parser.parse_args().results)
