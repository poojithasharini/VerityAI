import React,{useEffect,useState}from"react";import axios from"axios";import{ShieldCheck,BrainCircuit,Database,Activity,ScanSearch,Network,UploadCloud,ChartNoAxesCombined,Fingerprint,ChevronRight,AlertTriangle,CheckCircle2,Menu, X,ArrowUpRight,Zap}from"lucide-react";import{ResponsiveContainer,BarChart,Bar,XAxis,YAxis,Tooltip,CartesianGrid,PieChart,Pie,Cell,Legend}from"recharts";
const API="/api";

const initial = {
  dataset: {
    total: 696,
    genuine: 348,
    fake: 348
  },
  model_family: "Deep Learning",
  input_features: 16,
  fusion_dimension: 73,
  branches: [
    { name: "Feature Attention Network", output: "16-dimensional embedding and attention weights" },
    { name: "Genuine-only Autoencoder", output: "8-dimensional latent vector and anomaly score" },
    { name: "Graph Convolution Network", output: "16-dimensional graph embedding" },
    { name: "Hybrid DNN", output: "16-dimensional embedding" },
    { name: "Hybrid Feature Transformer", output: "16-dimensional embedding" }
  ],
  fusion: "73-dimensional representation followed by a 64-32-1 classifier",
  model_loaded: false
};

const pct=x=>(x*100).toFixed(2)+"%";
function App(){const[data,setData]=useState(initial),[page,setPage]=useState("Overview"),[online,setOnline]=useState(false),[form,setForm]=useState({profile_pic:1,username_digit_ratio:.08,fullname_words:2,fullname_digit_ratio:0,name_equals_username:0,description_length:85,external_url:0,private:0,posts:24,followers:420,follows:310}),[result,setResult]=useState(null),[file,setFile]=useState(null),[upload,setUpload]=useState(null),[busy,setBusy]=useState(false),[mobile,setMobile]=useState(false);
useEffect(()=>{let active=true;axios.get(API+"/health").then(r=>{if(active)setOnline(r.status===200);}).catch(err=>{console.error("Backend health check failed:",err.message,err.response?.data);if(active)setOnline(false);});axios.get(API+"/summary").then(r=>{if(active)setData(r.data);}).catch(err=>console.error("Summary request failed:",err.message,err.response?.data));return()=>{active=false;};},[]);
const nav=[["Overview",Activity],["Profile Analyzer",ScanSearch],["Dataset Lab",Database],["Model Performance",ChartNoAxesCombined],["Hybrid Architecture",BrainCircuit],["Explainability",Fingerprint]];
const predict = async (e) => {
  e.preventDefault();
  setBusy(true);
  setResult(null);
  try {
    const response = await axios.post(API + "/predict", form, {
      headers: { "Content-Type": "application/json" }
    });
    setResult(response.data);
  } catch (error) {
    console.error("Prediction API error:", error.response?.data || error);
    setResult({
      error: true,
      label: "Prediction unavailable",
      message: error.response?.data?.detail
        ? JSON.stringify(error.response.data.detail)
        : error.message
    });
  } finally {
    setBusy(false);
  }
};
const doUpload=async e=>{e.preventDefault();if(!file)return;setBusy(true);let f=new FormData();f.append("file",file);try{setUpload((await axios.post(API+"/upload",f)).data)}catch(err){setUpload({error:err.response?.data?.detail||"Could not connect to API"})}setBusy(false)};
const field=(key,label,type="number")=><label className="field"><span>{label}</span>{type==="select"?<select value={form[key]} onChange={e=>setForm({...form,[key]:+e.target.value})}><option value="0">No</option><option value="1">Yes</option></select>:<input type="number" value={form[key]} onChange={e=>setForm({...form,[key]:+e.target.value})}/>}</label>;
return <div className="shell"><aside className={mobile?"side open":"side"}><div className="brand"><div className="logo"><Fingerprint/></div><div><b>VERITY<span>AI</span></b><small>PROFILE INTELLIGENCE</small></div><button className="close" onClick={()=>setMobile(false)}><X/></button></div><div className="workspace">● &nbsp;Research workspace <span>⌄</span></div><small className="navcap">WORKSPACE</small>{nav.slice(0,3).map(([n,I])=><button key={n} className={page===n?"nav active":"nav"} onClick={()=>{setPage(n);setMobile(false)}}><I size={17}/>{n}</button>)}<small className="navcap">RESEARCH</small>{nav.slice(3).map(([n,I])=><button key={n} className={page===n?"nav active":"nav"} onClick={()=>{setPage(n);setMobile(false)}}><I size={17}/>{n}</button>)}<div className="sidefoot"><div className="status"><small>PROJECT STATUS</small><b>Research prototype <i>●</i></b><div className="track"><i/></div><span>Notebook experiments completed</span></div><div className="team"><div>DS</div><span><b>Data Science Team</b><small>SR University</small></span></div></div></aside>
<main><header><button className="hamb" onClick={()=>setMobile(true)}><Menu/></button><div className="crumb">Workspace　/　<b>{page}</b></div><div className="right"><span className={online?"online":"offline"}>● {online?"API CONNECTED":"API OFFLINE"}</span><div className="user">DS</div></div></header><section className="content">
{page==="Overview"&&<><div className="hero"><div className="kicker">● INTELLIGENT SOCIAL PROFILE ANALYTICS</div><h1>See beyond the<br/><em>profile.</em></h1><p>A hybrid intelligence framework for identifying suspicious social profiles through feature attention, anomaly detection and graph learning.</p><button className="cta" onClick={()=>setPage("Profile Analyzer")}> <ScanSearch size={16}/> Analyze a profile <ArrowUpRight size={15}/></button><div className="orb"><Fingerprint size={58}/><span>AI</span></div></div><div className="heading"><div><small>RESEARCH SNAPSHOT / 01</small><h2>Project at a glance</h2><p>Recorded experimental results from the Google Colab notebook.</p></div><b className="pill">EXPERIMENT COMPLETED</b></div>
<div className="metrics">
  {[
    [Database, "696", "Dataset profiles", "348 genuine · 348 fake"],
    [Activity, "16", "Input features", "Standardized profile features"],
    [BrainCircuit, "05", "DL branches", "Five complementary branches"],
    [Network, "73", "Fusion dimensions", "Combined branch representation"]
  ].map(([I, v, t, s]) => (
    <div className="metric" key={t}>
      <div><span>{t}</span><I /></div>
      <b>{v}</b>
      <small>{s}</small>
    </div>
  ))}
</div>

<div className="charts">
  <div className="panel">
    <h3>Five-branch deep learning architecture</h3>
    <p>Complementary representations combined by the fusion classifier.</p>
    {initial.branches.map((branch, i) => (
      <div className="evidence" key={branch.name}>
        <CheckCircle2 size={16} />
        <span><b>{String(i + 1).padStart(2, "0")} · {branch.name}</b><br />
        {branch.output}</span>
      </div>
    ))}
  </div>

  <div className="panel">
    <h3>Dataset composition</h3>
    <p>696 labelled profile records</p>
    <ResponsiveContainer width="100%" height={240}>
      <PieChart>
        <Pie
          data={[
            { name: "Genuine", value: 348 },
            { name: "Fake", value: 348 }
          ]}
          dataKey="value"
          innerRadius={62}
          outerRadius={88}
        >
          {[0, 1].map(x => (
            <Cell key={x} fill={x ? "#858bff" : "#42d9b6"} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  </div>
</div>

<div className="insight">
  <b>Research architecture</b>
  <p>
    VerityAI combines feature attention, genuine-profile anomaly detection,
    graph learning, a hybrid DNN and a feature transformer.
    Their representations are concatenated into a 73-dimensional vector
    for final classification.
  </p>
</div>
</>}

{page==="Profile Analyzer"&&<><div className="heading"><div><small>INTERACTIVE WORKSPACE / 02</small><h2>Profile analyzer</h2><p>Analyze profile characteristics using the trained hybrid deep learning model.</p>
</div></div><div className="two"><div className="panel"><h3>Profile attributes</h3><p>Enter the profile characteristics to generate an assessment using the trained hybrid model.</p><form onSubmit={predict} className="fields">{field("profile_pic","Profile picture present","select")}{field("username_digit_ratio","Username digit ratio")}{field("fullname_words","Full name words")}{field("fullname_digit_ratio","Full name digit ratio")}{field("name_equals_username","Name equals username","select")}{field("description_length","Bio description length")}{field("external_url","External URL","select")}{field("private","Private account","select")}{field("posts","Number of posts")}{field("followers","Number of followers")}{field("follows","Number of accounts followed")}<button className="cta wide" disabled={busy}><Zap size={16}/> {busy?"Analyzing...":"Analyze profile"}</button></form></div><div className="panel"><h3>Analysis output</h3>{!result?<div className="empty"><Fingerprint size={42}/><b>Awaiting profile input</b><p>Complete the profile attributes and select Analyze profile to view the model output.</p></div>:result.error?<div className="empty"><AlertTriangle size={32}/><b>{result.label}</b><p>{result.message}</p></div>:<div className="prediction"><b className="pill">{result.mode || "MODEL RESULT"}</b><div className="score">{(Number(result.score)*100).toFixed(1)}<small>%</small></div><h2>{result.label}</h2><p>Estimated fake-profile likelihood produced by the trained hybrid model.</p><div className="result-meta"><span>Risk level</span><strong>{result.risk_band || "Not available"}</strong></div><div className="result-meta"><span>Manual review</span><strong>{result.manual_review ? "Recommended" : "Not indicated"}</strong></div><h4>Signals identified</h4>{(result.evidence || []).map((x,i)=><div className="evidence" key={i}><CheckCircle2 size={15}/>{x}</div>)}<p className="subtle-note">This is a research model estimate based on the entered profile attributes, not independent verification of an Instagram account.</p></div>}</div></div></>}
{page==="Dataset Lab"&&<><div className="heading"><div><small>DATA MANAGEMENT / 03</small><h2>Dataset lab</h2><p>Upload a CSV to inspect its schema and data quality.</p></div></div><div className="two"><div className="panel"><h3><UploadCloud/> Upload dataset CSV</h3><p>Files are processed in memory and not stored permanently.</p><form onSubmit={doUpload}><label className="drop"><input type="file" accept=".csv" onChange={e=>setFile(e.target.files[0])}/><UploadCloud size={32}/><b>{file?file.name:"Choose a CSV file"}</b><small>Click to browse</small></label><button className="cta wide" disabled={!file||busy}>{busy?"Inspecting...":"Inspect CSV"}</button></form>{upload?.error&&<div className="warning">{upload.error}</div>}{upload&&!upload.error&&<div className="success">✓ {upload.message}</div>}</div><div className="panel"><h3>Notebook data audit</h3>{["11 original predictors + target","No missing values in supplied CSVs","Balanced 50 / 50 class distribution","2 duplicate rows in each CSV; retained","Follower count is highly skewed"].map((x,i)=><div className="evidence" key={i}><CheckCircle2 size={16}/>{x}</div>)}</div></div>{upload&&!upload.error&&<div className="panel"><h3>Uploaded file inspection</h3><div className="stats">{[["Rows",upload.rows],["Columns",upload.columns.length],["Missing cells",upload.missing_values],["Duplicates",upload.duplicate_rows]].map(([k,v])=><div><b>{v}</b><small>{k}</small></div>)}</div><div className="tablewrap"><table><thead><tr>{upload.columns.map(c=><th key={c}>{c}</th>)}</tr></thead><tbody>{upload.preview.map((r,i)=><tr key={i}>{upload.columns.map(c=><td key={c}>{String(r[c]??"—")}</td>)}</tr>)}</tbody></table></div></div>}</>}

{page === "Model Performance" && <>
  <div className="heading">
    <div>
      <small>DEEP LEARNING RESULTS / 04</small>
      <h2>Model performance</h2>
      <p>Evaluation workspace for the five-branch deep-learning model.</p>
    </div>
  </div>

  <div className="panel">
    <h3>Five-branch model configuration</h3>
    <div className="stats">
      <div><b>05</b><small>DL branches</small></div>
      <div><b>16</b><small>Input features</small></div>
      <div><b>73</b><small>Fusion dimensions</small></div>
      <div><b>64–32–1</b><small>Fusion classifier</small></div>
    </div>
  </div>

  <div className="panel">
    <h3>Deep-learning components</h3>
    {initial.branches.map((branch, i) => (
      <div className="evidence" key={branch.name}>
        <CheckCircle2 size={16}/>
        <span>
          <b>Branch {i + 1}: {branch.name}</b><br/>
          {branch.output}
        </span>
      </div>
    ))}
  </div>


</>}


{page === "Hybrid Architecture" && <>
  <div className="heading">
    <div>
      <small>MODEL DESIGN / 05</small>
      <h2>Hybrid deep learning architecture</h2>
      <p>Five complementary branches converge into a 73-dimensional fusion classifier.</p>
    </div>
  </div>

  <div className="archinput">
    INPUT · 16 STANDARDIZED PROFILE FEATURES
  </div>

  <div className="branches">
    {[
      ["01", "FEATURE ATTENTION NETWORK",
       "Learns the importance of individual profile features.",
       "16 input → attention scoring → feature weighting → 16→32→16 encoder",
       "16-D embedding + attention weights"],

      ["02", "GENUINE-ONLY AUTOENCODER",
       "Learns genuine-profile patterns and identifies unusual inputs.",
       "16→24→8 encoder → latent representation → 8→24→16 decoder",
       "8-D latent vector + anomaly score"],

      ["03", "GRAPH CONVOLUTION NETWORK",
       "Learns representations using neighbouring profiles in a similarity graph.",
       "KNN graph K=5 → two graph convolution layers → neighbour aggregation",
       "16-D graph embedding"],

      ["04", "HYBRID DNN",
       "Learns nonlinear patterns from the standardized profile features.",
       "16→64→32→16 encoder",
       "16-D embedding"],

      ["05", "HYBRID FEATURE TRANSFORMER",
       "Learns relationships between profile features using transformer layers.",
       "16 feature tokens → 32-D representation → transformer encoder → 16-D projection",
       "16-D embedding"]
    ].map(x => (
      <div className="panel branch" key={x[0]}>
        <small>BRANCH {x[0]}</small>
        <h3>{x[1]}</h3>
        <p>{x[2]}</p>
        <div>{x[3]}</div>
        <b>OUTPUT: {x[4]}</b>
      </div>
    ))}
  </div>

  <div className="fusion">
    <small>FIVE-BRANCH FUSION CLASSIFIER</small>
    <h2>16 + 8 + 1 + 16 + 16 + 16 = 73 dimensions</h2>
    <p>Dense layers: 73 → 64 → 32 → 1</p>
    <b>FINAL FAKE-PROFILE PROBABILITY</b>
  </div>


</>}


{page === "Explainability" && <>
  <div className="heading">
    <div>
      <small>MODEL INTERPRETABILITY / 06</small>
      <h2>Explainability studio</h2>
      <p>Inspect the signals produced by the five-branch deep-learning model.</p>
    </div>
  </div>

  <div className="two">
    <div className="panel">
      <h3>Feature attention</h3>
      <p>
        The Feature Attention Network assigns relative weights
        to the 16 standardized input features.
      </p>
      <div className="evidence">
        <BrainCircuit size={17}/>
        Attention weights are available for individual predictions
        through the prediction API.
      </div>
      <p className="subtle-note">
        Attention weights indicate model emphasis, not causal evidence.
      </p>
    </div>

    <div className="panel">
      <h3>Autoencoder anomaly detection</h3>
      <p>
        The genuine-only autoencoder learns to reconstruct
        genuine-profile patterns.
      </p>
      <div className="evidence">
        <Activity size={17}/>
        Reconstruction error is used as an anomaly signal.
      </div>
      <p className="subtle-note">
        A high reconstruction error does not independently prove
        that a profile is fake.
      </p>
    </div>
  </div>

  <div className="panel">
    <h3>Five-branch prediction evidence</h3>
    {[
      "Feature Attention Network — attention weights and embedding",
      "Genuine-only Autoencoder — latent representation and reconstruction error",
      "Graph Convolution Network — graph-based embedding",
      "Hybrid DNN — learned nonlinear representation",
      "Hybrid Feature Transformer — transformer-based representation"
    ].map((item, i) => (
      <div className="evidence" key={i}>
        <CheckCircle2 size={16}/>
        {item}
      </div>
    ))}
  </div>

  <div className="panel">
    <h3>Prediction-level explainability</h3>
    <p>
      Submit a profile in the Profile Analyzer to inspect its
      attention weights, anomaly score, branch probabilities,
      branch disagreement and nearest feature-space neighbours.
    </p>

  </div>
</>}

</section><footer>VERITYAI · RESEARCH PROTOTYPE <span>SR UNIVERSITY · CSE (DATA SCIENCE) · v1.0</span></footer></main></div>}
export default App;
