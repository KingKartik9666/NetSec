import {Fragment} from 'react';
export const percent = (value?: number | null) => typeof value === 'number' && Number.isFinite(value) ? `${(value*100).toFixed(2)}%` : 'Unavailable';
export const number = (value?: number | null, precision = 4) => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('en-US',{maximumFractionDigits:precision}) : 'Unavailable';
export function ErrorNotice({id, error}: {id: string; error?: string}) {return <div id={id} className="insight-error" role="alert" hidden={!error}>{error}</div>;}
export function Metadata({entries}: {entries: [string,unknown][]}) {return <dl className="evidence-meta">{entries.map(([key,value]) => <Fragment key={key}><dt>{key}</dt><dd>{typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? 'Unavailable')}</dd></Fragment>)}</dl>;}
export function Details({title,value}: {title: string; value: unknown}) {return <details><summary>{title}</summary><pre className="evidence-json">{JSON.stringify(value,null,2)}</pre></details>;}
export function Warnings({items = []}: {items?: unknown[]}) {return <ul className="warning-list">{items.map((item,index)=><li key={index}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>)}</ul>;}
export function download(name: string, content: string, type = 'application/json') {
  const url=URL.createObjectURL(new Blob([content],{type})); const anchor=document.createElement('a');
  anchor.href=url; anchor.download=name; anchor.hidden=true; document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
